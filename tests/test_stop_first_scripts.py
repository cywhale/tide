"""Tests for the stop-first Zarr maintenance helpers:
conf/pm2_app_lifecycle.sh, conf/dask_scheduler_check.sh, and
conf/py311_zarr_state.sh, plus the rollback option of py311_zarr_align.sh.

PM2 and ps are simulated: a fake pm2 keeps the app list and a fake process
table in PM2_HOME/state.json, and a fake ps reads the same table. Fake PIDs
start at 9,000,000, above any real PID limit, so `kill -0` treats them as
gone unless a test puts a real process into the table.
"""
import base64
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from test_ops_scripts import FAKE_PY, pm2_home  # noqa: F401  (fixture)

CONF = Path(__file__).resolve().parents[1] / "conf"
LIFE = CONF / "pm2_app_lifecycle.sh"
DASKC = CONF / "dask_scheduler_check.sh"
ZSTATE = CONF / "py311_zarr_state.sh"
ALIGN = CONF / "py311_zarr_align.sh"

PY311 = "/home/odbadmin/.pyenv/versions/py311/bin"
TIDE_ARGS = f"{PY311}/python3.11 {PY311}/gunicorn tide_app:app -w 2 -b 127.0.0.1:8040 --reload"
WORKER_ARGS = f"{PY311}/python3.11 {PY311}/dask-worker tls://127.0.0.1:8786 --host 127.0.0.1"
SPAWN_ARGS = f"{PY311}/python3.11 -c from multiprocessing.spawn import spawn_main"

# --- pm2_app_lifecycle.sh ----------------------------------------------------

FAKE_PM2 = r'''#!/usr/bin/env python3
import json, os, sys
home = os.environ["PM2_HOME"]
path = os.path.join(home, "state.json")
st = json.load(open(path))
with open(os.path.join(home, "calls.log"), "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
sc = st["scenario"]
apps = {a["name"]: a for a in st["apps"]}

def drop_tree(root):
    kids = {}
    for p in st["procs"]:
        kids.setdefault(p[1], []).append(p[0])
    gone, todo = set(), [root]
    while todo:
        q = todo.pop()
        gone.add(q)
        todo.extend(kids.get(q, []))
    keep = set(st.get("survivors", []))
    st["procs"] = [p for p in st["procs"] if p[0] not in gone or p[0] in keep]

cmd = sys.argv[1]
if cmd == "jlist":
    st["jlists"] = st.get("jlists", 0) + 1
    t = st.get("target")
    if t and st.get("acted"):
        a = apps[t]
        st["after"] = st.get("after", 0) + 1
        if sc == "respawn" and st["after"] >= 2 and a["status"] == "stopped":
            a.update(status="online", pid=a["pid"] or 9000900)
        if sc == "crashloop" and st["after"] >= 2:
            a["pid"] += 1
            a["restarts"] += 1
    def entry(a):   # pid "MISSING": PM2 5.4.2 omits the key for some stopped apps
        e = {"name": a["name"],
             "pm2_env": {"status": a["status"], "restart_time": a["restarts"]}}
        if a["pid"] != "MISSING":
            e["pid"] = a["pid"]
        return e
    print(json.dumps([entry(a) for a in st["apps"]]))
elif cmd in ("stop", "restart"):
    name = sys.argv[2]
    st["target"], st["acted"] = name, True
    if sc == cmd + "-fail":
        json.dump(st, open(path, "w")); sys.exit(1)
    a = apps[name]
    if sc == "other-changes":
        apps["gateway"]["pid"] += 1
    if cmd == "stop":
        if a["pid"]:
            if sc == "orphan":       # a child survives, reparented to init
                for p in st["procs"]:
                    if p[1] == a["pid"]:
                        p[1] = 1
                        st.setdefault("survivors", []).append(p[0])
            drop_tree(a["pid"])
        a.update(status="stopped", pid=0)
    else:
        if sc == "errored":
            a.update(status="errored", pid=0)
        else:
            a.update(status="online", pid=st["start_pid"])
            if sc != "never":
                st["procs"].extend(st["start_procs"])
            if sc == "dup":
                st["procs"].append([9000990, 1, st["start_procs"][0][2]])
            if sc == "count-up":
                a["restarts"] += 1
json.dump(st, open(path, "w"))
'''

FAKE_PS_TABLE = r'''#!/bin/bash
# ps -o args= -p PID | ps -eo pid=,ppid=,args=
pid=""; all=no
while [ $# -gt 0 ]; do
  case "$1" in -p) pid=$2; shift 2 ;; -eo) all=yes; shift 2 ;; *) shift ;; esac
done
if [ -n "$pid" ] && [ "$pid" = "${FAKE_DAEMON_PID:-}" ]; then
  echo "$FAKE_DAEMON_ARGS"; exit 0
fi
python3 -I - "$FAKE_STATE" "$all" "$pid" <<'PY'
import json, sys
st = json.load(open(sys.argv[1]))
if sys.argv[2] == "yes":
    for p in st["procs"]:
        print("%s %s %s" % tuple(p))
else:
    for p in st["procs"]:
        if str(p[0]) == sys.argv[3]:
            print(p[2]); sys.exit(0)
    sys.exit(1)
PY
'''


def _world(home, scenario, target, status, pid, procs, start_pid=0, start_procs=(),
           extra=()):
    apps = [{"name": "gateway", "status": "online", "pid": 9000007, "restarts": 0},
            {"name": target, "status": status, "pid": pid, "restarts": 3}]
    apps += [dict(a) for a in extra]
    (home / "state.json").write_text(json.dumps({
        "scenario": scenario, "apps": apps, "procs": [list(p) for p in procs],
        "start_pid": start_pid, "start_procs": [list(p) for p in start_procs]}))


def _calls(home):
    log = home / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


@pytest.fixture
def life(tmp_path, pm2_home):  # noqa: F811
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "pm2").write_text(FAKE_PM2)
    (bindir / "ps").write_text(FAKE_PS_TABLE)
    for f in ("pm2", "ps"):
        (bindir / f).chmod(0o755)

    def run(mode, name, scenario="ok", status="online", pid=9000100, procs=(),
            start_pid=9000200, start_procs=(), match="tide_app:app",
            expect_exec=f"{PY311}/gunicorn", chain=True, home=None, extra=()):
        if home is None:
            home, dpid = pm2_home()
        else:
            home, dpid = home
        _world(home, scenario, name, status, pid, procs, start_pid, start_procs, extra)
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}",
                   FAKE_STATE=str(home / "state.json"), FAKE_DAEMON_PID=str(dpid),
                   FAKE_DAEMON_ARGS=f"PM2 v5.4.2: God Daemon ({home})")
        env.pop("PM2_HOME", None)
        cmd = (f'"{LIFE}" {mode} --pm2 "{bindir / "pm2"}" --pm2-home "{home}" '
               f'--name {name} --timeout 2 --settle 1')
        if mode in ("stop", "start"):
            cmd += f' --match {match}'
            if expect_exec:
                cmd += f' --expect-exec "{expect_exec}"'
        if chain:
            cmd += " && echo NEXT_STEP_WOULD_RUN"
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                           env=env, timeout=60)
        return r, _calls(home), json.loads((home / "state.json").read_text())
    return run


TIDE_TREE = [(9000100, 1, TIDE_ARGS), (9000101, 9000100, TIDE_ARGS),
             (9000102, 9000100, TIDE_ARGS)]
TIDE_NEW = [(9000200, 1, TIDE_ARGS), (9000201, 9000200, TIDE_ARGS),
            (9000202, 9000200, TIDE_ARGS)]


def test_stop_passes_and_uses_only_pm2_stop(life):
    r, calls, st = life("stop", "tide", procs=TIDE_TREE)
    assert r.returncode == 0, r.stderr
    assert "OK stop: tide pid 9000100 stopped" in r.stdout
    assert "NEXT_STEP_WOULD_RUN" in r.stdout
    assert [c for c in calls if c != "jlist"] == ["stop tide"]
    assert st["procs"] == []


def test_stop_of_an_errored_app_brings_it_to_stopped(life):
    r, calls, _ = life("stop", "tide", status="errored", pid=0)
    assert r.returncode == 0, r.stderr
    assert "stop tide" in calls


@pytest.mark.parametrize("scenario, status, pid, procs, why", [
    ("ok", "stopped", 0, [], "already stopped"),
    ("ok", "online", 9000100, TIDE_TREE + [(9000300, 1, TIDE_ARGS)],
     "expected one process tree"),                        # duplicate before the stop
    ("ok", "online", 9000100, [(9000100, 1, "/usr/bin/python3 gunicorn tide_app:app")],
     "expected /home/odbadmin"),                          # wrong executable
    ("stop-fail", "online", 9000100, TIDE_TREE, "pm2 stop tide exited non-zero"),
    ("respawn", "online", 9000100, TIDE_TREE, "PM2 restarted it"),
    ("orphan", "online", 9000100, TIDE_TREE, "still run after the stop"),
    ("other-changes", "online", 9000100, TIDE_TREE, "another PM2 app changed"),
])
def test_stop_fails_closed(life, scenario, status, pid, procs, why):
    r, calls, _ = life("stop", "tide", scenario=scenario, status=status, pid=pid, procs=procs)
    assert r.returncode != 0
    assert why in r.stderr
    assert "NEXT_STEP_WOULD_RUN" not in r.stdout
    if why in ("already stopped", "expected one process tree", "expected /home/odbadmin"):
        assert not any(c.startswith("stop") for c in calls)   # refused before pm2 stop


def test_stop_waits_for_every_process_of_the_tree(life):
    # A real process in the tree that outlives the PM2 stop: the check must
    # time out rather than report the app stopped.
    p = subprocess.Popen(["sleep", "30"])
    try:
        tree = TIDE_TREE + [(p.pid, 9000100, "sleep 30")]
        r, _, _ = life("stop", "tide", scenario="orphan", procs=tree)
        assert r.returncode != 0
        assert "still running:" in r.stderr and str(p.pid) in r.stderr
    finally:
        p.kill()
        p.wait()


def test_start_passes_with_pm2_restart_and_no_update_env(life):
    r, calls, _ = life("start", "tide", status="stopped", pid=0, start_procs=TIDE_NEW)
    assert r.returncode == 0, r.stderr
    assert "OK start: tide online as pid 9000200" in r.stdout
    assert [c for c in calls if c != "jlist"] == ["restart tide"]
    assert not any("--update-env" in c or c.startswith("start") for c in calls)


def test_start_accepts_an_errored_app(life):
    r, _, _ = life("start", "tide", status="errored", pid=0, start_procs=TIDE_NEW)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("scenario, status, procs, start_procs, why", [
    ("ok", "online", TIDE_TREE, TIDE_NEW, "not stopped"),
    ("ok", "stopped", [(9000300, 1, TIDE_ARGS)], TIDE_NEW, "already run before the start"),
    ("restart-fail", "stopped", [], TIDE_NEW, "pm2 restart tide exited non-zero"),
    ("errored", "stopped", [], TIDE_NEW, "not online with a matching process"),
    ("never", "stopped", [], TIDE_NEW, "not online with a matching process"),
    ("crashloop", "stopped", [], TIDE_NEW, "crash loop"),
    ("count-up", "stopped", [], TIDE_NEW, "restart count went from 3 to 4"),
    ("dup", "stopped", [], TIDE_NEW, "expected one process tree"),
    ("ok", "stopped", [], [(9000200, 1, f"/usr/bin/python3 /usr/bin/gunicorn tide_app:app")],
     "expected /home/odbadmin"),
    ("other-changes", "stopped", [], TIDE_NEW, "another PM2 app changed"),
])
def test_start_fails_closed(life, scenario, status, procs, start_procs, why):
    pid = 9000100 if status == "online" else 0
    r, calls, _ = life("start", "tide", scenario=scenario, status=status, pid=pid,
                       procs=procs, start_procs=start_procs)
    assert r.returncode != 0
    assert why in r.stderr
    assert "NEXT_STEP_WOULD_RUN" not in r.stdout
    if why in ("not stopped", "already run before the start"):
        assert not any(c.startswith("restart") for c in calls)


def test_nanny_worker_tree_counts_as_one_app(life):
    # The nanny is PM2's PID; its spawned worker and resource tracker do not
    # carry the dask-worker word, so the nanny is the only match root.
    tree = [(9000100, 1, WORKER_ARGS), (9000101, 9000100, SPAWN_ARGS),
            (9000102, 9000100, f"{PY311}/python3.11 -c from multiprocessing.resource_tracker")]
    r, _, st = life("stop", "dask-worker", procs=tree, match=f"{PY311}/dask-worker",
                    expect_exec=f"{PY311}/dask-worker")
    assert r.returncode == 0, r.stderr
    assert st["procs"] == []


def test_operator_command_line_does_not_count_as_a_match(life, tmp_path):
    # The checker's own command line holds the --match word; it must not be
    # taken for a second copy of the app (a real ps, not the fake table).
    real_ps = subprocess.run(["bash", "-c", "command -v ps"], capture_output=True,
                             text=True).stdout.strip()
    text = LIFE.read_text()
    start = text.index("match_roots() {")
    func = text[start:text.index("\n}\n", start) + 3]
    probe = tmp_path / "probe.sh"
    probe.write_text(func + 'proc_table() { ' + real_ps + ' -eo pid=,ppid=,args=; }\n'
                     'MATCH=zz_unique_word_zz\nmatch_roots\n')
    r = subprocess.run(["bash", str(probe), "zz_unique_word_zz"], capture_output=True,
                       text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""


def test_lifecycle_never_calls_pm2_without_a_live_daemon(life, pm2_home):  # noqa: F811
    home, _ = pm2_home(sock="stale")
    r, calls, _ = life("stop", "tide", procs=TIDE_TREE, home=(home, None))
    assert r.returncode != 0 and "refusing to call pm2" in r.stderr
    assert calls == []


def test_status_and_pid_modes_are_read_only(life):
    r, calls, _ = life("status", "tide", procs=TIDE_TREE, chain=False)
    assert r.returncode == 0 and r.stdout.strip() == "online 9000100 3"
    assert calls == ["jlist"]
    r, calls, _ = life("pid", "tide", procs=TIDE_TREE, chain=False)
    assert r.returncode == 0 and r.stdout.strip() == "9000100"
    r, _, _ = life("pid", "tide", status="stopped", pid=0, chain=False)
    assert r.returncode != 0 and "not online" in r.stderr


# VM24, 2026-10-29: the stopped app odbbathy has no pid in `pm2 jlist`
# (PM2 5.4.2 loads a stopped app from its dump with `process: {}`).
STOPPED_NO_PID = {"name": "odbbathy", "status": "stopped", "pid": "MISSING", "restarts": 0}


@pytest.mark.parametrize("pid", ["MISSING", None])
def test_a_stopped_app_without_a_pid_reads_as_pid_0(life, pid):
    r, calls, _ = life("status", "tide", status="stopped", pid=pid, chain=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "stopped 0 3"
    assert calls == ["jlist"]


def test_a_stopped_bystander_without_a_pid_does_not_block_any_mode(life):
    r, _, _ = life("status", "tide", procs=TIDE_TREE, chain=False, extra=[STOPPED_NO_PID])
    assert r.returncode == 0 and r.stdout.strip() == "online 9000100 3", r.stderr
    r, _, _ = life("pid", "tide", procs=TIDE_TREE, chain=False, extra=[STOPPED_NO_PID])
    assert r.returncode == 0 and r.stdout.strip() == "9000100", r.stderr
    r, calls, _ = life("stop", "tide", procs=TIDE_TREE, extra=[STOPPED_NO_PID])
    assert r.returncode == 0 and "NEXT_STEP_WOULD_RUN" in r.stdout, r.stderr
    r, calls, _ = life("start", "tide", status="stopped", pid=0, start_procs=TIDE_NEW,
                       extra=[STOPPED_NO_PID])
    assert r.returncode == 0 and "NEXT_STEP_WOULD_RUN" in r.stdout, r.stderr


def test_start_of_a_stopped_app_without_a_pid(life):
    r, calls, _ = life("start", "tide", status="stopped", pid="MISSING", start_procs=TIDE_NEW)
    assert r.returncode == 0, r.stderr
    assert "OK start: tide online as pid 9000200" in r.stdout
    assert [c for c in calls if c != "jlist"] == ["restart tide"]


# Without a positive PID, an online app's executable and process tree cannot
# be checked; any other status without a pid is not a state PM2 5.4.2
# produces. Each makes every mode fail before pm2 stop or restart.
BAD_PIDS = [("online", "MISSING"), ("online", None), ("online", 0), ("online", -5),
            ("online", "9000100"), ("online", True), ("online", 9000100.0),
            ("errored", "MISSING"), ("errored", None), ("launching", "MISSING"),
            ("stopping", None)]


@pytest.mark.parametrize("mode", ["status", "pid", "stop", "start"])
@pytest.mark.parametrize("status, pid", BAD_PIDS)
def test_a_target_without_a_valid_pid_fails_closed(life, mode, status, pid):
    r, calls, _ = life(mode, "tide", status=status, pid=pid, procs=TIDE_TREE,
                       start_procs=TIDE_NEW)
    assert r.returncode != 0
    assert "unexpected PM2 entry 'tide'" in r.stderr
    assert "cannot read PM2 state" in r.stderr or "cannot read the PM2 app list" in r.stderr
    assert "NEXT_STEP_WOULD_RUN" not in r.stdout
    assert calls == ["jlist"]                       # refused before any pm2 action


@pytest.mark.parametrize("status, pid", [("online", "MISSING"), ("errored", None)])
def test_a_bystander_without_a_valid_pid_fails_closed(life, status, pid):
    bad = {"name": "ghrsst", "status": status, "pid": pid, "restarts": 7}
    r, calls, _ = life("stop", "tide", procs=TIDE_TREE, extra=[bad])
    assert r.returncode != 0
    assert "unexpected PM2 entry 'ghrsst'" in r.stderr
    assert not any(c.startswith("stop") for c in calls)


def test_malformed_restart_count_fails_closed(life):
    bad = {"name": "ghrsst", "status": "online", "pid": 9000500, "restarts": None}
    r, calls, _ = life("status", "tide", procs=TIDE_TREE, chain=False, extra=[bad])
    assert r.returncode != 0 and "restart_time None" in r.stderr


# --- dask_scheduler_check.sh -------------------------------------------------

def _metrics(clients=0, tasks=None, workers=1):
    tasks = tasks or {}
    lines = [f"dask_scheduler_clients {clients}.0"]
    for state in ("released", "waiting", "no-worker", "queued", "processing",
                  "memory", "erred"):
        lines.append(f'dask_scheduler_tasks{{state="{state}"}} {tasks.get(state, 0)}.0')
    lines.append(f'dask_scheduler_workers{{state="idle"}} {workers}.0')
    return "\n".join(lines) + "\n"


def _identity(workers):
    return json.dumps({"type": "Scheduler", "workers": {
        f"tls://127.0.0.1:{4000 + i}": {"status": st, "nanny": nanny}
        for i, (st, nanny) in enumerate(workers)}})


FAKE_CURL_WEB = r'''#!/bin/bash
echo "$*" >> "$WEB/curl.log"
url=${@: -1}
path=${url#http://127.0.0.1:8787}
f="$WEB/$(echo "$path" | tr '/' '_')"
[ -f "$f" ] || { echo "curl: (7) Failed to connect" >&2; exit 7; }
cat "$f"
'''


@pytest.fixture
def daskc(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "curl").write_text(FAKE_CURL_WEB)
    (bindir / "ss").write_text('#!/bin/bash\necho "$*" >> "$WEB/ss.log"\n'
                               'cat "$WEB/ss.out" 2>/dev/null\n')
    for f in ("curl", "ss"):
        (bindir / f).chmod(0o755)

    def serve(health="ok", metrics=None, identity=None, ss_pid=None):
        for name, body in (("_health", health), ("_metrics", metrics or _metrics()),
                           ("_json_identity.json", identity or _identity([("running", "tls://127.0.0.1:46037")]))):
            p = web / name
            if body is None:
                p.unlink(missing_ok=True)
            else:
                p.write_text(body)
        if ss_pid is not None:
            (web / "ss.out").write_text(
                f'LISTEN 0 2048 127.0.0.1:46037 0.0.0.0:* users:(("dask-worker",pid={ss_pid},fd=7))\n')

    def run(args, chain=True):
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", WEB=str(web))
        cmd = f'"{DASKC}" {args}' + (" && echo NEXT_STEP_WOULD_RUN" if chain else "")
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                              env=env, timeout=60)
    return serve, run, web


def test_drained_passes_with_no_clients_and_no_tasks(daskc):
    serve, run, web = daskc
    serve()
    r = run("drained --timeout 1")
    assert r.returncode == 0, r.stderr
    assert "OK drained: clients=0 tasks=0" in r.stdout
    calls = (web / "curl.log").read_text().splitlines()
    assert all("--proto =http" in c and "http://127.0.0.1:8787/" in c for c in calls)
    assert not any(t in c for c in calls for t in (" -X", " -d", "--data", " -k", "--insecure"))


@pytest.mark.parametrize("kw, why", [
    ({"metrics": _metrics(clients=1)}, "clients=1"),
    ({"metrics": _metrics(tasks={"processing": 2})}, "tasks=2"),
    ({"metrics": _metrics(tasks={"memory": 1, "waiting": 1})}, "tasks=2"),
    ({"health": None}, "not reachable"),
    ({"metrics": "dask_scheduler_workers{state=\"idle\"} 1.0\n"}, "metrics lack"),
])
def test_drained_fails_closed(daskc, kw, why):
    serve, run, _ = daskc
    serve(**kw)
    r = run("drained --timeout 1")
    assert r.returncode != 0
    assert "drained not reached" in r.stderr and why in r.stderr
    assert "NEXT_STEP_WOULD_RUN" not in r.stdout


def test_ready_passes_with_the_expected_nanny(daskc):
    serve, run, web = daskc
    serve(metrics=_metrics(clients=4), ss_pid=101867)
    r = run("ready --workers 1 --nanny-pid 101867 --clients 4 --timeout 1")
    assert r.returncode == 0, r.stderr
    assert "nanny PID 101867 owns the nanny port" in r.stdout
    assert "sport = :46037" in (web / "ss.log").read_text()


@pytest.mark.parametrize("kw, args, why", [
    ({"identity": _identity([])}, "--workers 1", "workers=0"),
    ({"identity": _identity([("running", "tls://127.0.0.1:46037")] * 2)}, "--workers 1", "workers=2"),
    ({"identity": _identity([("closing", "tls://127.0.0.1:46037")])}, "--workers 1", "status=closing"),
    ({"metrics": _metrics(clients=3)}, "--workers 1 --clients 4", "clients=3"),
    ({"ss_pid": 555}, "--workers 1 --nanny-pid 101867", "owned by [555]"),
    ({"identity": _identity([("running", None)]), "ss_pid": 101867},
     "--workers 1 --nanny-pid 101867", "reports no nanny"),
])
def test_ready_fails_closed(daskc, kw, args, why):
    serve, run, _ = daskc
    serve(**kw)
    r = run(f"ready {args} --timeout 1")
    assert r.returncode != 0
    assert why in r.stderr
    assert "NEXT_STEP_WOULD_RUN" not in r.stdout


def test_dask_check_refuses_a_non_loopback_url(daskc):
    serve, run, web = daskc
    serve()
    for url in ("https://127.0.0.1:8787", "http://10.0.0.5:8787", "http://localhost:8787"):
        r = run(f"drained --url {url} --timeout 1")
        assert r.returncode != 0 and "--url must be" in r.stderr
    assert not (web / "curl.log").exists()


# --- py311_zarr_state.sh -----------------------------------------------------

def _rec(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _wheel(out: Path, version: str, extra=None) -> Path:
    files = {"zarr/__init__.py": f'__version__ = "{version}"\n'.encode(),
             "zarr/core.py": f"x = {version!r}\n".encode()}
    files.update(extra or {})
    di = f"zarr-{version}.dist-info"
    files[f"{di}/METADATA"] = f"Metadata-Version: 2.1\nName: zarr\nVersion: {version}\n".encode()
    files[f"{di}/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    lines = [f"{k},{_rec(v)},{len(v)}" for k, v in files.items()] + [f"{di}/RECORD,,"]
    files[f"{di}/RECORD"] = ("\n".join(lines) + "\n").encode()
    w = out / f"zarr-{version}-py3-none-any.whl"
    with zipfile.ZipFile(w, "w") as z:
        for k, v in files.items():
            z.writestr(k, v)
    return w


@pytest.fixture(scope="module")
def venv_base(tmp_path_factory):
    """A real venv with pip (bundled, offline) and hand-built zarr wheels."""
    base = tmp_path_factory.mktemp("zstate")
    subprocess.run([sys.executable, "-m", "venv", str(base / "venv")], check=True,
                   capture_output=True, timeout=300)
    wheels = base / "wheels"
    wheels.mkdir()
    _wheel(wheels, "2.18.6")
    _wheel(wheels, "2.18.7", {"zarr/new_in_7.py": b"y = 1\n"})
    return base


@pytest.fixture
def zenv(venv_base, tmp_path):
    import shutil
    venv = tmp_path / "venv"
    shutil.copytree(venv_base / "venv", venv, symlinks=True)
    py = venv / "bin" / "python"
    wheels = venv_base / "wheels"

    def pip(*args):
        return subprocess.run([str(py), "-I", "-m", "pip", "--disable-pip-version-check", *args],
                              capture_output=True, text=True, timeout=300)

    def install(version, *opts):
        r = pip("install", "--no-deps", "--no-index", "--find-links", str(wheels), *opts,
                f"zarr=={version}")
        assert r.returncode == 0, r.stderr

    install("2.18.6")
    d = tmp_path / "align"
    d.mkdir()
    freeze = pip("freeze", "--all").stdout
    (d / "freeze-before.txt").write_text(
        "\n".join(sorted(freeze.splitlines(), key=lambda s: s.lower())) + "\n")
    check = pip("check")
    (d / "pip-check-before.txt").write_text(check.stdout + f"exit {check.returncode}\n")
    site = Path(subprocess.run([str(py), "-I", "-c",
                                "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                               capture_output=True, text=True).stdout.strip())

    def state(expect=None):
        cmd = [str(ZSTATE), "--dir", str(d), "--python", str(py)]
        if expect:
            cmd += ["--expect", expect]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return state, install, pip, site, d


def test_state_before_and_after_are_recognised(zenv):
    state, install, _, _, _ = zenv
    r = state("before")
    assert r.returncode == 0 and r.stdout.strip() == "STATE before", r.stdout + r.stderr
    install("2.18.7")
    r = state("after")
    assert r.returncode == 0 and r.stdout.strip() == "STATE after", r.stdout + r.stderr
    r = state("before")
    assert r.returncode != 0 and "expected before" in r.stderr


@pytest.mark.parametrize("damage, why", [
    ("hash", "hash mismatch zarr/core.py"),
    ("missing", "missing zarr/core.py"),
    ("stray", "not in RECORD zarr/left_over.py"),
    ("extra-package", "freeze matches neither state"),
])
def test_state_unknown_when_the_package_is_not_clean(zenv, damage, why):
    state, install, pip, site, _ = zenv
    if damage == "hash":
        (site / "zarr" / "core.py").write_text("x = 'damaged'\n")
    elif damage == "missing":
        (site / "zarr" / "core.py").unlink()
    elif damage == "stray":
        (site / "zarr" / "left_over.py").write_text("z = 1\n")
    else:
        (site / "other_pkg-1.0.dist-info").mkdir()
        (site / "other_pkg-1.0.dist-info" / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: other-pkg\nVersion: 1.0\n")
    r = state()
    assert r.returncode != 0
    assert "STATE unknown" in r.stdout and why in r.stdout


def test_state_file_check_ignores_dist_info_without_metadata(zenv):
    # py311 on VM24 holds dist-info directories with only REQUESTED. The zarr
    # file check must neither crash on them nor count a zarr one as a second
    # install. (pip 24 in this venv warns about them on stderr, which the
    # pip check comparison rejects; pip 25.3 on VM24 did not warn.)
    state, _, _, site, _ = zenv
    for name in ("oldpkg-1.0.dist-info", "zarr-2.18.3.dist-info"):
        (site / name).mkdir()
        (site / name / "REQUESTED").write_text("")
    r = state()
    assert "zarr files:" not in r.stdout, r.stdout


def test_state_unknown_with_two_installed_zarr_dist_infos(zenv):
    state, _, _, site, _ = zenv
    extra = site / "zarr-2.18.7.dist-info"
    extra.mkdir()
    (extra / "METADATA").write_text("Metadata-Version: 2.1\nName: zarr\nVersion: 2.18.7\n")
    r = state()
    assert r.returncode != 0
    assert "expected one zarr dist-info with METADATA, found 2" in r.stdout


def test_state_unknown_when_zarr_does_not_import(zenv):
    state, _, _, site, _ = zenv
    (site / "zarr" / "__init__.py").write_text("raise ImportError('half-installed')\n")
    r = state()
    assert r.returncode != 0
    assert "zarr does not import" in r.stdout


def test_rollback_without_force_reinstall_does_not_repair(zenv):
    # Why the aligner's rollback uses --force-reinstall: pip reports a
    # damaged 2.18.6 with intact metadata as already satisfied.
    state, install, _, site, _ = zenv
    (site / "zarr" / "core.py").write_text("x = 'damaged'\n")
    install("2.18.6")
    assert "hash mismatch" in state().stdout
    install("2.18.6", "--force-reinstall")
    r = state("before")
    assert r.returncode == 0, r.stdout


def test_state_writes_nothing(zenv):
    state, _, _, _, d = zenv
    before = sorted(p.name for p in d.iterdir())
    state()
    assert sorted(p.name for p in d.iterdir()) == before


# --- py311_zarr_align.sh rollback --------------------------------------------

def test_aligner_rollback_force_reinstalls_and_apply_does_not(tmp_path):
    fake = tmp_path / "fakepy"
    fake.mkdir()
    (fake / "zarr").write_text("2.18.6\n")
    py = fake / "python"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    env = dict(os.environ, FAKE_PY=str(fake), FAKE_FAIL="", FAKE_CHECK="conflicts")
    d = tmp_path / "align"
    for mode in ("prepare", "apply", "rollback"):
        r = subprocess.run(["bash", str(ALIGN), mode, "--dir", str(d), "--python", str(py)],
                           capture_output=True, text=True, env=env, timeout=60)
        assert r.returncode == 0, (mode, r.stderr)
    installs = [c for c in (fake / "calls").read_text().splitlines()
                if c.startswith("install") and "--dry-run" not in c]
    assert len(installs) == 2
    assert "--force-reinstall" not in installs[0] and "zarr-2.18.7.req" in installs[0]
    assert "--force-reinstall" in installs[1] and "zarr-2.18.6.req" in installs[1]


# --- py311_quiescent.sh ------------------------------------------------------

QUIESCE = CONF / "py311_quiescent.sh"
P311 = "/home/odbadmin/.pyenv/versions/py311/"


@pytest.fixture
def fakeproc(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()

    def add(pid, cmdline=b"/usr/sbin/sshd\0-D\0", maps="7f00 r-xp 0 00:00 0 /usr/lib/libc.so\n",
            uid=1000, readable=True):
        d = proc / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(cmdline)
        (d / "maps").write_text(maps)
        (d / "status").write_text(f"Name:\tx\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
        if not readable:
            (d / "maps").chmod(0)
        return d

    def run(*args):
        return subprocess.run([str(QUIESCE), "--proc", str(proc), *args],
                              capture_output=True, text=True, timeout=60)
    yield proc, add, run
    for m in proc.glob("*/maps"):
        m.chmod(0o644)


def test_quiescent_passes_when_nothing_uses_py311(fakeproc):
    _, add, run = fakeproc
    add(1, b"/sbin/init\0")
    add(200)
    r = run()
    assert r.returncode == 0, r.stderr
    assert "OK quiescent: none of 2 live processes" in r.stdout


@pytest.mark.parametrize("where", ["cmdline", "maps"])
def test_quiescent_fails_on_a_py311_process(fakeproc, where):
    _, add, run = fakeproc
    add(1)
    if where == "cmdline":
        add(4242, (P311 + "bin/python3.11\0x.py\0").encode())
    else:   # started through the pyenv shim: argv0 is just "python"
        add(4242, b"python\0x.py\0",
            maps=f"7f00 r-xp 0 00:00 0 {P311}lib/python3.11/site-packages/numpy/_m.so\n")
    r = run()
    assert r.returncode != 0
    assert "processes use py311" in r.stderr and "4242 uid=1000" in r.stderr
    assert "OK quiescent" not in r.stdout


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads every file")
def test_quiescent_fails_closed_on_a_live_process_it_cannot_read(fakeproc):
    # Regression: an unreadable /proc/<pid>/maps used to be counted and
    # passed; it may belong to a process that uses py311.
    _, add, run = fakeproc
    add(1)
    add(77, b"/usr/sbin/nginx\0", uid=0, readable=False)
    r = run()
    assert r.returncode != 0
    assert "1 live processes could not be inspected" in r.stderr
    assert "77 uid=0 /usr/sbin/nginx (PermissionError)" in r.stderr
    assert "OK" not in r.stdout
    r = run("--inspectable-only")
    assert r.returncode != 0 and "could not be inspected" in r.stderr


def test_quiescent_skips_a_process_that_exited_meanwhile(fakeproc):
    # Listed by os.listdir, gone by the time it is read: only a vanished
    # /proc entry may be ignored.
    proc, add, run = fakeproc
    add(1)
    (proc / "999").symlink_to(proc / "no-such-dir")
    r = run()
    assert r.returncode == 0, r.stderr
    assert "none of 1 live processes" in r.stdout


def test_quiescent_inspectable_only_ignores_py311_use(fakeproc):
    _, add, run = fakeproc
    add(4242, (P311 + "bin/python3.11\0").encode())
    r = run("--inspectable-only")
    assert r.returncode == 0, r.stderr
    assert "OK inspectable: all 1 live processes could be read" in r.stdout
