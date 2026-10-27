"""v0.3.2 operator scripts: atomic manifest replacement, HTTPS smoke, and the
verified PM2 restart.

All three run for real against sandboxes (a temp release directory and a
temp git checkout; `curl`, `pm2`, `ps`, `stat`, and `mv` stubs on PATH), so
the fail-fast control flow is exercised, not just the syntax.
"""
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPLACE = REPO_ROOT / "conf" / "replace_runtime_files.sh"
SMOKE = REPO_ROOT / "conf" / "https_smoke.sh"
MANIFEST = ("conf/start_app.sh", "src/dask_client_manager.py")
OLD = {"conf/start_app.sh": "#!/bin/bash\necho old launcher\n",
       "src/dask_client_manager.py": "OLD_CLIENT = True\n"}
NEW = {"conf/start_app.sh": "#!/bin/bash\necho new launcher\n",
       "src/dask_client_manager.py": "NEW_CLIENT = True\n"}
OTHER = {"tide_app.py": "app = 1\n", "__init__.py": "",
         "src/store_adapter.py": "ADAPTER = 1\n",
         "conf/ecosystem.config.js": "module.exports = {};\n"}


def _write(root: Path, files: dict):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def _git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                           *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def sb(tmp_path):
    rel = tmp_path / "release"
    _write(rel, {**OLD, **OTHER})
    (rel / "conf/start_app.sh").chmod(0o755)
    src = tmp_path / "checkout"
    _write(src, NEW)
    _git(src, "init", "-q")
    _git(src, "add", ".")
    _git(src, "commit", "-q", "-m", "merged")
    return {"tmp": tmp_path, "rel": rel, "src": src,
            "commit": _git(src, "rev-parse", "HEAD"),
            "work": tmp_path / "ops" / "deploy-1"}


def _replace(sb, mode, path_prefix=None, env_extra=None, **over):
    args = {"rel": sb["rel"], "work": sb["work"]}
    if mode == "deploy":
        args.update(src=sb["src"], commit=sb["commit"])
    args.update(over)
    cmd = ["bash", str(REPLACE), mode]
    for k, v in args.items():
        cmd += [f"--{k}", str(v)]
    env = dict(os.environ, **(env_extra or {}))
    if path_prefix:
        env["PATH"] = f"{path_prefix}:{env['PATH']}"
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)


def _stub(dirpath: Path, name: str, body: str) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_text(body)
    p.chmod(0o755)
    return dirpath


def _contents(root):
    return {rel: (root / rel).read_text() for rel in (*MANIFEST, *OTHER)}


def test_deploy_replaces_manifest_by_rename_and_keeps_the_rest(sb):
    before_inodes = {f: (sb["rel"] / f).stat().st_ino for f in MANIFEST}
    r = _replace(sb, "deploy")
    assert r.returncode == 0, r.stderr
    for f in MANIFEST:
        assert (sb["rel"] / f).read_text() == NEW[f]
        # a rename installs a new inode; an in-place copy would keep the old one
        assert (sb["rel"] / f).stat().st_ino != before_inodes[f]
        assert (sb["work"] / "backup" / f).read_text() == OLD[f]
    for f, text in OTHER.items():
        assert (sb["rel"] / f).read_text() == text
    assert stat.S_IMODE((sb["rel"] / "conf/start_app.sh").stat().st_mode) == 0o755
    assert (sb["work"] / "source-commit").read_text().strip() == sb["commit"]
    assert not (sb["work"] / "staging").exists()


def test_deploy_refuses_existing_work_dir(sb):
    (sb["work"] / "backup").mkdir(parents=True)
    marker = sb["work"] / "backup" / "earlier.txt"
    marker.write_text("keep me")
    before = _contents(sb["rel"])
    r = _replace(sb, "deploy")
    assert r.returncode != 0
    assert "already exists" in r.stderr
    assert _contents(sb["rel"]) == before
    assert marker.read_text() == "keep me"


def test_deploy_hash_mismatch_replaces_nothing(sb):
    # The second manifest file differs from the commit: the first must not be
    # replaced either, because all files are verified before any rename.
    (sb["src"] / "src/dask_client_manager.py").write_text("TAMPERED = True\n")
    before = _contents(sb["rel"])
    r = _replace(sb, "deploy")
    assert r.returncode != 0
    assert "hash mismatch" in r.stderr and "nothing was replaced" in r.stderr
    assert _contents(sb["rel"]) == before


def test_deploy_refuses_wrong_commit(sb):
    before = _contents(sb["rel"])
    r = _replace(sb, "deploy", commit="0123456789abcdef")
    assert r.returncode != 0
    assert "not 0123456789abcdef" in r.stderr
    assert _contents(sb["rel"]) == before


def test_deploy_refuses_cross_filesystem_staging(sb, tmp_path):
    # Stub `stat` reports a different device for the work directory.
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    stub = stub_dir / "stat"
    stub.write_text('#!/bin/bash\nfor a in "$@"; do case "$a" in *xdevwork*) '
                    'echo 111; exit 0;; esac; done\nexec /usr/bin/stat "$@"\n')
    stub.chmod(0o755)
    before = _contents(sb["rel"])
    r = _replace(sb, "deploy", path_prefix=stub_dir, work=tmp_path / "xdevwork" / "w")
    assert r.returncode != 0
    assert "would not be atomic" in r.stderr
    assert _contents(sb["rel"]) == before


def test_restore_puts_back_backups(sb):
    assert _replace(sb, "deploy").returncode == 0
    r = _replace(sb, "restore")
    assert r.returncode == 0, r.stderr
    for f in MANIFEST:
        assert (sb["rel"] / f).read_text() == OLD[f]
    assert stat.S_IMODE((sb["rel"] / "conf/start_app.sh").stat().st_mode) == 0o755


# Second `mv` (the src/dask_client_manager.py swap) fails outright, or
# "succeeds" but leaves wrong content, after conf/start_app.sh was replaced.
STUB_MV = r'''#!/bin/bash
n=$(( $(cat "$MV_COUNT" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$MV_COUNT"
if [ "$n" -eq 2 ]; then
  case "$MV_MODE" in
    fail) echo "mv: simulated failure" >&2; exit 1 ;;
    corrupt) for a in "$@"; do last=$a; done; printf 'CORRUPT\n' > "$last"; exit 0 ;;
  esac
fi
exec /bin/mv "$@"
'''


@pytest.mark.parametrize("mv_mode", ["fail", "corrupt"])
def test_partial_replacement_failure_is_recoverable_by_restore(sb, tmp_path, mv_mode):
    stub_dir = _stub(tmp_path / "mvstub", "mv", STUB_MV)
    r = _replace(sb, "deploy", path_prefix=stub_dir,
                 env_extra={"MV_COUNT": str(tmp_path / "mv.count"), "MV_MODE": mv_mode})
    assert r.returncode != 0
    assert "restore --rel" in r.stderr          # the error names the recovery
    # Mixed state: the launcher was replaced, the client was not (or is bad).
    assert (sb["rel"] / "conf/start_app.sh").read_text() == NEW["conf/start_app.sh"]
    assert (sb["rel"] / "src/dask_client_manager.py").read_text() != NEW["src/dask_client_manager.py"]

    r = _replace(sb, "restore")                 # normal mv again
    assert r.returncode == 0, r.stderr
    for f in MANIFEST:
        assert (sb["rel"] / f).read_text() == OLD[f]
    assert stat.S_IMODE((sb["rel"] / "conf/start_app.sh").stat().st_mode) == 0o755


def _record_lines(sb):
    return (sb["work"] / "backup.sha256").read_text().splitlines()


@pytest.mark.parametrize("case,expect", [
    ("truncated", "missing"),
    ("duplicate", "duplicate"),
    ("traversal", "unexpected path"),
    ("extra", "unexpected path"),
    ("malformed", "malformed"),
    ("empty", "missing"),
])
def test_restore_rejects_bad_backup_record(sb, tmp_path, case, expect):
    assert _replace(sb, "deploy").returncode == 0
    lines = _record_lines(sb)
    h0 = lines[0].split()[0]
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n")
    record = {
        "truncated": lines[:1],
        "duplicate": [lines[0], lines[0]],
        "traversal": [lines[0], f"{h0}  ../../outside.txt"],
        "extra": [*lines, f"{h0}  src/store_adapter.py"],
        "malformed": [lines[0], lines[1][:20] + lines[1][64:]],
        "empty": [],
    }[case]
    (sb["work"] / "backup.sha256").write_text("".join(f"{x}\n" for x in record))
    r = _replace(sb, "restore")
    assert r.returncode != 0
    assert expect in r.stderr
    for f in MANIFEST:                            # nothing restored, not even partly
        assert (sb["rel"] / f).read_text() == NEW[f]
    assert outside.read_text() == "outside\n"


def test_restore_refuses_tampered_backup(sb):
    assert _replace(sb, "deploy").returncode == 0
    (sb["work"] / "backup/src/dask_client_manager.py").write_text("BAD = 1\n")
    r = _replace(sb, "restore")
    assert r.returncode != 0
    assert "hash mismatch" in r.stderr
    for f in MANIFEST:
        assert (sb["rel"] / f).read_text() == NEW[f]


# --- https_smoke.sh -------------------------------------------------------

FAKE_CURL = r'''#!/bin/bash
if [ "$1" = "--version" ]; then
  echo "curl ${FAKE_CURL_VERSION:-8.5.0} (fake)"
  echo "Release-Date: fake"
  exit 0
fi
echo "$*" >> "$CURL_LOG"
out=""; url=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out=$2; shift 2 ;;
    -w|--resolve) shift 2 ;;
    -*) shift ;;
    *) url=$1; shift ;;
  esac
done
port=${url#https://*:}; port=${port%%/*}
code_var=FAKE_${port}_CODE; body_var=FAKE_${port}_BODY
code=${!code_var:-200}; body=${!body_var:-}
printf '%s' "$body" > "$out"
printf '%s' "$code"
[ "$code" -lt 400 ] || exit 22
'''


@pytest.fixture
def smoke(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "curl").write_text(FAKE_CURL)
    (bindir / "curl").chmod(0o755)
    log = tmp_path / "curl.log"

    def run(mode, curl_version="8.5.0", **resp):
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", CURL_LOG=str(log),
                   FAKE_CURL_VERSION=curl_version)
        for port, (code, body) in resp.items():
            env[f"FAKE_{port[1:]}_CODE"] = str(code)
            env[f"FAKE_{port[1:]}_BODY"] = body
        r = subprocess.run(["bash", str(SMOKE), "--host", "eco.odb.ntu.edu.tw",
                            "--out-root", str(tmp_path / "smoke-root"), "--mode", mode],
                           capture_output=True, text=True, env=env, timeout=30)
        calls = log.read_text().splitlines() if log.exists() else []
        return r, calls
    return run


BODY_A = '{"z":[1.0,2.0]}'
BODY_B = '{"z":[1.5,2.5]}'


def test_smoke_same_passes_on_identical_200(smoke):
    r, calls = smoke("same", p8041=(200, BODY_A), p8040=(200, BODY_A))
    assert r.returncode == 0, r.stderr
    assert len(calls) == 2
    for c, port in zip(calls, ("8041", "8040")):
        assert "--fail-with-body" in c
        assert f"--resolve eco.odb.ntu.edu.tw:{port}:127.0.0.1" in c
        assert f"https://eco.odb.ntu.edu.tw:{port}/api/tide?" in c
        assert " -k" not in f" {c}" and "--insecure" not in c


def test_smoke_same_fails_on_different_bodies(smoke):
    r, _ = smoke("same", p8041=(200, BODY_A), p8040=(200, BODY_B))
    assert r.returncode != 0 and "differ" in r.stderr


def test_smoke_different_passes_and_fails(smoke):
    r, _ = smoke("different", p8041=(200, BODY_B), p8040=(200, BODY_A))
    assert r.returncode == 0, r.stderr
    r, _ = smoke("different", p8041=(200, BODY_A), p8040=(200, BODY_A))
    assert r.returncode != 0 and "identical" in r.stderr


def test_smoke_identical_error_responses_fail(smoke):
    # Two equal 500 bodies must not pass as "identical".
    r, calls = smoke("same", p8041=(500, "oops"), p8040=(500, "oops"))
    assert r.returncode != 0
    assert "port 8041" in r.stderr
    assert len(calls) == 1          # stops at the first failure


def test_smoke_non_200_success_code_fails(smoke):
    r, _ = smoke("same", p8041=(204, BODY_A), p8040=(200, BODY_A))
    assert r.returncode != 0 and "expected 200" in r.stderr


def test_smoke_empty_object_fails(smoke):
    r, _ = smoke("same", p8041=(200, "{}"), p8040=(200, "{}"))
    assert r.returncode != 0 and "{}" in r.stderr


def test_smoke_prod_mode(smoke):
    r, calls = smoke("prod", p8040=(200, BODY_A))
    assert r.returncode == 0, r.stderr
    assert len(calls) == 1 and ":8040/" in calls[0]
    r, _ = smoke("prod", p8040=(503, "down"))
    assert r.returncode != 0


def test_smoke_each_attempt_gets_a_new_directory(smoke, tmp_path):
    # A failed attempt followed by an immediate (same-second) retry: the retry
    # is not blocked and the failed attempt's output is kept.
    r1, _ = smoke("same", p8041=(500, "oops"), p8040=(200, BODY_A))
    r2, _ = smoke("same", p8041=(200, BODY_A), p8040=(200, BODY_A))
    assert r1.returncode != 0
    assert r2.returncode == 0, r2.stderr
    d1 = Path(re.search(r"^attempt: (.+)$", r1.stdout, re.M).group(1))
    d2 = Path(re.search(r"^attempt: (.+)$", r2.stdout, re.M).group(1))
    assert d1 != d2 and d1.parent == d2.parent == tmp_path / "smoke-root"
    assert (d1 / "canary.json").read_text() == "oops"
    assert (d2 / "prod.json").read_text() == BODY_A


@pytest.mark.parametrize("version", ["7.68.0", "7.75.9", "6.99.0", "garbage"])
def test_smoke_refuses_old_or_unknown_curl_before_any_request(smoke, version):
    r, calls = smoke("same", curl_version=version, p8041=(200, BODY_A), p8040=(200, BODY_A))
    assert r.returncode != 0
    assert "7.76" in r.stderr
    assert calls == []


@pytest.mark.parametrize("version", ["7.76.0", "7.88.1", "8.5.0"])
def test_smoke_accepts_curl_7_76_and_later(smoke, version):
    r, _ = smoke("same", curl_version=version, p8041=(200, BODY_A), p8040=(200, BODY_A))
    assert r.returncode == 0, r.stderr


# --- pm2_restart_check.sh -------------------------------------------------

RESTART = REPO_ROOT / "conf" / "pm2_restart_check.sh"
PY311_GUNICORN = "/home/odbadmin/.pyenv/versions/py311/bin/gunicorn"
GOOD_ARGS = ("/home/odbadmin/.pyenv/versions/py311/bin/python3.11 "
             f"{PY311_GUNICORN} tide_app:app -w 2 -b 127.0.0.1:8040")

# Each fake daemon is a PM2_HOME directory holding state.json and calls.log;
# the fake pm2 acts on whichever PM2_HOME it is given (KeyError if none).
# `restart` changes the state per scenario, and in "flap" the app keeps
# restarting after the first post-restart poll (crash loop).
FAKE_PM2 = r'''#!/usr/bin/env python3
import json, os, sys
home = os.environ["PM2_HOME"]
path = os.path.join(home, "state.json")
with open(os.path.join(home, "calls.log"), "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
st = json.load(open(path))
if sys.argv[1] == "jlist":
    if st["restarted"]:
        st["post"] += 1
        if st["mode"] == "flap" and st["post"] >= 2:
            st["pid"] += 1
            st["restarts"] += 1
    apps = [{"name": "gateway", "pid": 7, "pm2_env": {"status": "online", "restart_time": 0}}]
    if st["mode"] != "missing":
        apps.append({"name": "tide", "pid": st["pid"],
                     "pm2_env": {"status": st["status"], "restart_time": st["restarts"]}})
    print(json.dumps(apps))
elif sys.argv[1] == "restart":
    if st["mode"] == "fail":
        sys.exit(1)
    st["restarted"] = True
    st["pid"] += {"samepid": 0}.get(st["mode"], 100)
    st["restarts"] += {"double": 2}.get(st["mode"], 1)
    if st["mode"] == "errored":
        st["status"] = "errored"
    if st["mode"] == "respawn":     # a different daemon took over mid-run
        with open(os.path.join(home, "pm2.pid"), "w") as f:
            f.write(str(os.getppid()))
json.dump(st, open(path, "w"))
'''


def _calls(home: Path):
    log = home / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def _state(home: Path, mode: str):
    (home / "state.json").write_text(json.dumps({
        "mode": mode, "pid": 4242, "restarts": 3, "status": "online",
        "restarted": False, "post": 0}))


# A stand-in PM2 daemon: a real process listening on rpc.sock.
LISTENER = ("import socket, sys\n"
            "s = socket.socket(socket.AF_UNIX); s.bind(sys.argv[1]); s.listen(8)\n"
            "while True:\n    c, _ = s.accept(); c.close()\n")


@pytest.fixture
def pm2_home():
    """Factory for PM2_HOME dirs. Short paths under /tmp keep rpc.sock within
    the Unix-socket path limit. sock: live | stale | file | none;
    pid: daemon (the listener) | dead | other (a live non-daemon) | none."""
    made, procs = [], []

    def make(mode=None, sock="live", pid="daemon"):
        home = Path(tempfile.mkdtemp(prefix="pm2t-", dir="/tmp"))
        made.append(home)
        if mode:
            _state(home, mode)
        listener = None
        if sock == "live":
            listener = subprocess.Popen([sys.executable, "-c", LISTENER, str(home / "rpc.sock")])
            procs.append(listener)
            deadline = time.time() + 5
            while not (home / "rpc.sock").exists():
                assert time.time() < deadline, "listener did not bind"
                time.sleep(0.05)
        elif sock == "stale":           # a real socket file nobody listens on
            s = socket.socket(socket.AF_UNIX)
            s.bind(str(home / "rpc.sock"))
            s.close()
        elif sock == "file":
            (home / "rpc.sock").write_text("")
        daemon_pid = None
        if pid == "daemon":
            if listener is None:        # a live process that claims the title
                listener = subprocess.Popen(["sleep", "60"])
                procs.append(listener)
            daemon_pid = listener.pid
            (home / "pm2.pid").write_text(f"{daemon_pid}\n")
        elif pid == "dead":
            p = subprocess.Popen(["true"])
            p.wait()
            (home / "pm2.pid").write_text(f"{p.pid}\n")
        elif pid == "other":
            p = subprocess.Popen(["sleep", "60"])
            procs.append(p)
            (home / "pm2.pid").write_text(f"{p.pid}\n")
        return home, daemon_pid

    yield make
    for p in procs:
        p.kill()
        p.wait()
    for h in made:
        shutil.rmtree(h, ignore_errors=True)


@pytest.fixture
def restart(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "pm2").write_text(FAKE_PM2)
    (bindir / "pm2").chmod(0o755)
    # ps: the stand-in daemon's PID (and a swap-in daemon's) shows the daemon
    # title, anything else the Gunicorn command line under test. With
    # FAKE_SWAP_TO, the first look at the original daemon rewrites pm2.pid to
    # the swap-in PID: a daemon replaced right after the pre-check verified it.
    (bindir / "ps").write_text(r'''#!/bin/bash
pid=""
while [ $# -gt 0 ]; do case "$1" in -p) pid=$2; shift 2 ;; *) shift ;; esac; done
if [ -n "${FAKE_SWAP_TO:-}" ] && [ "$pid" = "${FAKE_DAEMON_PID:-}" ] && [ ! -e "$FAKE_SWAP_MARK" ]; then
  echo "$FAKE_SWAP_TO" > "$FAKE_PIDFILE"
  : > "$FAKE_SWAP_MARK"
fi
if [ -n "${FAKE_DAEMON_PID:-}" ] && { [ "$pid" = "$FAKE_DAEMON_PID" ] || [ "$pid" = "${FAKE_SWAP_TO:-}" ]; }; then
  echo "$FAKE_DAEMON_ARGS"
else
  echo "$FAKE_PS_ARGS"
fi
''')
    (bindir / "ps").chmod(0o755)

    def run(home, daemon_pid=None, ps_args=GOOD_ARGS, chain=False,
            inherited_home=None, daemon_title=None, swap_to=None, extra=""):
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", FAKE_PS_ARGS=ps_args,
                   FAKE_DAEMON_ARGS=daemon_title or f"PM2 v5.4.2: God Daemon ({home})")
        env.pop("PM2_HOME", None)
        if daemon_pid is not None:
            env["FAKE_DAEMON_PID"] = str(daemon_pid)
        if swap_to is not None:
            env.update(FAKE_SWAP_TO=str(swap_to), FAKE_PIDFILE=str(home / "pm2.pid"),
                       FAKE_SWAP_MARK=str(tmp_path / "swap.mark"))
        if inherited_home is not None:
            env["PM2_HOME"] = str(inherited_home)
        cmd = (f'"{RESTART}" --pm2 "{bindir / "pm2"}" --pm2-home "{home}" '
               f"--timeout 2 --settle 1 {extra}")
        if chain:
            cmd += " && echo SMOKE_WOULD_RUN"
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                           env=env, timeout=60)
        return r, _calls(home)
    return run


def test_restart_check_passes_on_effective_restart(restart, pm2_home):
    home, dpid = pm2_home("ok")
    r, calls = restart(home, dpid, chain=True)
    assert r.returncode == 0, r.stderr
    assert "OK restart: tide pid 4242 -> 4342, restarts 3 -> 4" in r.stdout
    assert "SMOKE_WOULD_RUN" in r.stdout
    assert "restart tide" in calls          # stored environment only
    assert not any("--update-env" in c for c in calls)


@pytest.mark.parametrize("mode,ps_args,expect", [
    ("fail", GOOD_ARGS, "exited non-zero"),
    ("samepid", GOOD_ARGS, "not online with a new PID"),     # old process still serving
    ("errored", GOOD_ARGS, "not online with a new PID"),
    ("double", GOOD_ARGS, "expected exactly +1"),
    ("flap", GOOD_ARGS, "not stable"),
    ("ok", "/usr/bin/python3 /usr/local/bin/gunicorn tide_app:app", "expected " + PY311_GUNICORN),
    ("missing", GOOD_ARGS, "cannot read PM2 state"),
    ("respawn", GOOD_ARGS, "changed during the run"),        # race detected, not prevented
])
def test_restart_check_fails_and_blocks_chained_smoke(restart, pm2_home, mode, ps_args, expect):
    home, dpid = pm2_home(mode)
    r, _ = restart(home, dpid, ps_args=ps_args, chain=True)
    assert r.returncode != 0
    assert expect in r.stderr
    assert "SMOKE_WOULD_RUN" not in r.stdout
    assert "OK restart" not in r.stdout


def test_restart_check_ignores_inherited_pm2_home(restart, pm2_home):
    # The shell still points at a test daemon that also has a "tide" app; the
    # checker must restart the --pm2-home daemon and leave the decoy alone.
    home, dpid = pm2_home("ok")
    decoy, _ = pm2_home("ok")
    decoy_state = (decoy / "state.json").read_text()
    r, calls = restart(home, dpid, inherited_home=decoy)
    assert r.returncode == 0, r.stderr
    assert "restart tide" in calls
    assert f"daemon: PM2_HOME={home}" in r.stdout
    assert "ignoring inherited PM2_HOME" in r.stderr
    assert _calls(decoy) == []
    assert (decoy / "state.json").read_text() == decoy_state


@pytest.mark.parametrize("sock,pid,title,expect", [
    ("none", "none", None, "not a socket"),
    ("file", "daemon", None, "not a socket"),                # plain file named rpc.sock
    ("stale", "daemon", None, "refuses connections"),        # socket left by a dead daemon
    ("live", "none", None, "pm2.pid is missing"),
    ("live", "dead", None, "is not running"),
    ("live", "other", None, "is not the PM2 daemon"),        # PID reused by another process
    ("live", "daemon", "PM2 v5.4.2: God Daemon (/home/someone/.pm2)",
     "is not the PM2 daemon"),                               # another home's daemon
])
def test_restart_check_refuses_without_live_daemon_and_never_calls_pm2(
        restart, pm2_home, sock, pid, title, expect):
    home, dpid = pm2_home("ok", sock=sock, pid=pid)
    r, calls = restart(home, dpid, chain=True, daemon_title=title)
    assert r.returncode != 0
    assert "no running PM2 daemon" in r.stderr and expect in r.stderr
    assert calls == []                      # pm2 CLI never invoked: no new daemon
    assert "SMOKE_WOULD_RUN" not in r.stdout


def test_restart_check_detects_daemon_replaced_right_after_precheck(restart, pm2_home):
    # pm2.pid is rewritten to another live daemon the moment the pre-check has
    # verified the original one. The baseline must be the verified PID, so the
    # end-of-run comparison catches the swap.
    home, dpid = pm2_home("ok")
    other = subprocess.Popen(["sleep", "60"])
    try:
        r, calls = restart(home, dpid, chain=True, swap_to=other.pid)
    finally:
        other.kill()
        other.wait()
    assert r.returncode != 0
    assert f"daemon alive: PID {dpid} at {home}" in r.stdout
    assert f"changed during the run (PID {dpid} -> {other.pid})" in r.stderr
    assert "OK restart" not in r.stdout and "SMOKE_WOULD_RUN" not in r.stdout


def test_daemon_only_reports_live_daemon_without_calling_pm2(restart, pm2_home):
    home, dpid = pm2_home("ok")
    r, calls = restart(home, dpid, extra="--daemon-only")
    assert r.returncode == 0, r.stderr
    assert f"daemon alive: PID {dpid} at {home}" in r.stdout
    assert "OK daemon-only: pm2 was not called" in r.stdout
    assert calls == []


@pytest.mark.parametrize("sock,pid", [("stale", "daemon"), ("live", "dead"), ("none", "none")])
def test_daemon_only_fails_closed_without_calling_pm2(restart, pm2_home, sock, pid):
    home, dpid = pm2_home("ok", sock=sock, pid=pid)
    r, calls = restart(home, dpid, extra="--daemon-only")
    assert r.returncode != 0
    assert "no running PM2 daemon" in r.stderr
    assert calls == []


def test_ops_scripts_syntax():
    for script in (REPLACE, SMOKE, RESTART):
        subprocess.run(["bash", "-n", str(script)], check=True)
