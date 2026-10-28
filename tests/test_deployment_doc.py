"""Mechanical checks on the VM24 runbook (dev_tpxo10/DEPLOYMENT_VM24.md).

These pin the commands that were validated on VM24 (2026-10-27): shell blocks
must parse, production smoke comparisons go through the public 443 frontend
(the internal 8040 listener serves an expired certificate), TLS verification
is never disabled, and PM2 dummy configs under /home/odbadmin/tmp are .cjs
(that tree is in ES module scope).
"""
import re
import subprocess
import textwrap
import time
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "dev_tpxo10" / "DEPLOYMENT_VM24.md"
TEXT = DOC.read_text()
BLOCKS = [textwrap.dedent(body) for _, body in
          re.findall(r"^( *)```bash\n(.*?)^\1```", TEXT, flags=re.S | re.M)]


def _commands(block: str):
    """Shell commands with backslash continuations joined."""
    return [c.strip() for c in block.replace("\\\n", " ").splitlines() if c.strip()]


ALL_COMMANDS = [c for b in BLOCKS for c in _commands(b)]


def test_runbook_has_shell_blocks():
    assert len(BLOCKS) >= 8


@pytest.mark.parametrize("i", range(len(BLOCKS)))
def test_shell_blocks_parse(i, tmp_path):
    script = tmp_path / f"block{i}.sh"
    script.write_text(BLOCKS[i])
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_smoke_invocations_compare_through_443():
    smoke = [c for c in ALL_COMMANDS
             if ("https_smoke.sh" in c or '"$SMOKE"' in c) and "--mode" in c]
    assert len(smoke) >= 3          # TPXO10 canary, TPXO9 canary, post-deploy
    for c in smoke:
        assert "--prod-port 443" in c, c
        assert "--prod-port 8040" not in c, c


def test_no_tls_verification_bypass_in_commands():
    for c in ALL_COMMANDS:
        if re.search(r"\bcurl\b", c):
            assert not re.search(r"(^|\s)(-k|--insecure)(\s|$)", c), c


def test_no_angle_bracket_placeholders_in_commands():
    # `<name>` pasted into a shell is a redirection, not a placeholder.
    for c in ALL_COMMANDS:
        assert not re.search(r"<[A-Za-z][\w.-]*>", c), c


CANARY = [b for b in BLOCKS if "gunicorn tide_app:app" in b]


def test_canary_start_is_a_runnable_block():
    assert len(CANARY) == 1
    block = CANARY[0]
    c = [c for c in _commands(block) if "gunicorn tide_app:app" in c][0]
    assert '--keyfile "$LE_PRIVKEY" --certfile "$LE_FULLCHAIN"' in c, c
    assert '--chdir "$CHECKOUT"' in c, c
    assert "-b 127.0.0.1:8041" in c, c
    assert "--reload" not in c, c
    # Fail-closed preamble, scoped to a subshell so the operator's shell
    # keeps its own options and gets none of the exports.
    assert block.startswith("CANARY_PID=$(\n  set -euo pipefail\n"), block
    for needle in ('"${SHA:?', '"${LE_PRIVKEY:?', '"${LE_FULLCHAIN:?',
                   'cd "$CHECKOUT"', "git rev-parse HEAD",
                   '[[ $head == "$SHA"* ]]', "git diff --quiet HEAD --"):
        assert needle in block, needle
        assert block.index(needle) < block.index("gunicorn tide_app:app")


@pytest.fixture
def canary_sandbox(tmp_path):
    """The canary block with host paths moved into tmp_path and a stub
    Gunicorn that records its cwd and argv; returns (run, sha, repo,
    marker)."""
    home = tmp_path / "home"
    tls = home / "tls"
    for f in ("certs/ca.crt", "certs/tide-client.crt",
              "private/tide-client.key"):
        (tls / f).parent.mkdir(parents=True, exist_ok=True)
        (tls / f).write_text("x")
    (home / "store").mkdir()
    for f in ("le-key", "le-chain"):
        (home / f).write_text("x")
    marker = tmp_path / "gunicorn-called"
    stub = home / "gunicorn"
    stub.write_text(f'#!/bin/bash\necho "$PWD $*" > "{marker}"\n')
    stub.chmod(0o755)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(git + ["init", "-q", str(repo)], check=True)
    (repo / "tide_app.py").write_text("")
    subprocess.run(git + ["-C", str(repo), "add", "."], check=True)
    subprocess.run(git + ["-C", str(repo), "commit", "-qm", "c"], check=True)
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True
                         ).stdout.strip()
    (home / "tmp").mkdir()
    (home / "tmp" / f"tide-v0.3.2-{sha[:12]}").symlink_to(repo)
    script = CANARY[0]
    for old, new in (
            ("/home/odbadmin/.config/dask/tls", str(tls)),
            ("/home/odbadmin/python/tide_tpxo10_atlas_v2/data/tpxo10.zarr",
             str(home / "store")),
            ("/home/odbadmin/.pyenv/versions/py311/bin/gunicorn", str(stub)),
            ("/home/odbadmin/tmp", str(home / "tmp"))):
        assert old in script, old
        script = script.replace(old, new)

    def run(**env):
        base = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                "SHA": sha[:12], "LE_PRIVKEY": str(home / "le-key"),
                "LE_FULLCHAIN": str(home / "le-chain")}
        base.update(env)
        base = {k: v for k, v in base.items() if v is not None}
        return subprocess.run(["bash", "-c", script], env=base,
                              capture_output=True, text=True,
                              cwd=tmp_path)
    return run, sha, repo, marker


def test_canary_block_starts_only_from_the_expected_checkout(canary_sandbox):
    run, sha, repo, marker = canary_sandbox
    r = run()
    assert r.returncode == 0, r.stderr
    assert "canary PID" in r.stdout
    # The stub runs in the background of the block's subshell; give it time.
    for _ in range(50):
        if marker.exists() and marker.read_text().endswith("\n"):
            break
        time.sleep(0.1)
    cwd, args = marker.read_text().split(" ", 1)
    assert Path(cwd).resolve() == repo.resolve()
    argv = args.split()
    assert Path(argv[argv.index("--chdir") + 1]).resolve() == repo.resolve()
    assert "-b 127.0.0.1:8041" in args


@pytest.mark.parametrize("env, why", [
    ({"SHA": None}, "SHA"),
    ({"SHA": ""}, "SHA"),
    ({"SHA": "not-hex"}, "hex"),
    ({"SHA": "abcdef1"}, "tide-v0.3.2-abcdef1"),   # no checkout: cd fails
    ({"LE_PRIVKEY": None}, "LE_PRIVKEY"),
    ({"LE_FULLCHAIN": "/nonexistent"}, "not readable"),
])
def test_canary_block_fails_closed(canary_sandbox, env, why):
    run, _, _, marker = canary_sandbox
    r = run(**env)
    assert r.returncode != 0
    assert why in r.stderr
    assert not marker.exists()


def test_canary_block_refuses_wrong_head_or_dirty_checkout(canary_sandbox):
    run, sha, repo, marker = canary_sandbox
    (repo / "tide_app.py").write_text("changed")
    r = run()
    assert r.returncode != 0 and "tracked changes" in r.stderr
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["commit", "-qam", "d"], check=True)
    r = run()
    assert r.returncode != 0 and "not " + sha[:12] in r.stderr
    assert not marker.exists()


def test_pm2_dummy_config_is_cjs():
    assert "envprobe.config.cjs" in TEXT
    assert not re.search(r"envprobe\.config\.js\b", TEXT)


def test_rollback_status_keeps_production_caveat():
    status = re.search(r"^## Rollback\n\nStatus: (.*?)\n\n", TEXT,
                       flags=re.S | re.M)
    assert status, "Rollback section must open with a Status line"
    line = " ".join(status.group(1).split())
    if "components validated" in line:
        assert "production rollback not exercised" in line, line


def test_canary_does_not_use_expired_release_certificate():
    assert "<release>/conf/privkey.pem" not in TEXT
    assert "<release>/conf/fullchain.pem" not in TEXT


# --- Py311 Zarr alignment chains ---------------------------------------------

ALIGN_SCRIPT = DOC.parents[1] / "conf" / "py311_zarr_align.sh"
STUB_STEP = '#!/bin/bash\necho "$(basename "$0") $*" >> "$STEPS_LOG"\n'


def _align_block(mode):
    blocks = [b for b in BLOCKS if f'"$ALIGN" {mode} --dir "$D"' in b]
    assert len(blocks) == 1, mode
    return blocks[0]


MHW_LAUNCHER = "/home/odbadmin/proj/marineheatwave/API/conf/start_app.sh"


def test_align_blocks_chain_restart_and_smoke_with_and():
    assert len([b for b in BLOCKS if '"$ALIGN" prepare --dir "$D"' in b]) == 1
    for mode, out in (("apply", "after"), ("rollback", "rollback")):
        cmds = _commands(_align_block(mode))
        assert len(cmds) == 1, cmds            # one && chain, nothing after it
        steps = [s.strip() for s in cmds[0].split("&&")]
        assert len(steps) == 5, steps
        assert steps[0] == f'"$ALIGN" {mode} --dir "$D"'
        assert "pm2_restart_check.sh" in steps[1] and "--name tide" in steps[1]
        assert "--update-env" not in cmds[0]
        assert "https_smoke.sh" in steps[2] and "--prod-port 443" in steps[2]
        assert "pm2_restart_check.sh" in steps[3]
        assert f"--name mhwapi --expect-exec {MHW_LAUNCHER}" in " ".join(steps[3].split())
        assert steps[4] == f'mhw_check "$D-mhw-{out}" "$D-mhw-before"'


def test_run_sheet_reuses_the_prepared_directory():
    sets = [c for c in ALL_COMMANDS if c.startswith("D=")]
    assert "D=/home/odbadmin/tmp/zarr-align-20261028T103416" in sets
    # The only other D= is the prepare block for a new attempt.
    assert len(sets) == 2, sets


@pytest.fixture
def align_chain(tmp_path):
    from test_ops_scripts import FAKE_PY
    state = tmp_path / "fakepy"
    state.mkdir()
    (state / "zarr").write_text("2.18.6\n")
    py = state / "python"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    wrapper = tmp_path / "align"
    wrapper.write_text(f'#!/bin/bash\nexec bash "{ALIGN_SCRIPT}" "$@" --python "{py}"\n')
    wrapper.chmod(0o755)
    src = tmp_path / "src"
    for name in ("pm2_restart_check.sh", "https_smoke.sh"):
        p = src / "conf" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(STUB_STEP)
        p.chmod(0o755)
    log = tmp_path / "steps.log"
    d = tmp_path / "zarr-align-x"

    # The run sheet defines mhw_check in an earlier block of the same shell.
    mhw_stub = 'mhw_check() { echo "mhw_check $*" >> "$STEPS_LOG"; }\n'

    def run(block, fail="", check="conflicts", dirpath=d):
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "SRC": str(src),
               "ALIGN": str(wrapper), "D": str(dirpath), "FAKE_PY": str(state),
               "FAKE_FAIL": fail, "FAKE_CHECK": check, "STEPS_LOG": str(log)}
        return subprocess.run(["bash", "-c", mhw_stub + block], env=env,
                              cwd=tmp_path, capture_output=True, text=True,
                              timeout=60)

    r = run('"$ALIGN" prepare --dir "$D"')
    assert r.returncode == 0, r.stderr
    return run, log, state


def _steps(log):
    return log.read_text().splitlines() if log.exists() else []


@pytest.mark.parametrize("fail", ["install", "install-numpy"])
def test_align_apply_chain_stops_before_restart_and_smoke(align_chain, fail):
    run, log, _ = align_chain
    r = run(_align_block("apply"), fail)
    assert r.returncode != 0
    assert _steps(log) == []                     # no restart, no smoke


def test_align_apply_chain_stops_when_verification_freeze_fails(align_chain):
    run, log, state = align_chain
    fake = (state / "python").read_text()
    fake = fake.replace('echo "$v" > "$S/zarr"; echo "Successfully',
                        'echo "$v" > "$S/zarr"; touch "$S/after"; echo "Successfully')
    fake = fake.replace("  freeze)\n", '  freeze)\n    [ -e "$S/after" ] && exit 1\n')
    (state / "python").write_text(fake)
    r = run(_align_block("apply"))
    assert r.returncode != 0 and _steps(log) == []


def test_align_apply_then_rollback_chains_run_restart_and_smoke(align_chain):
    run, log, state = align_chain
    order = ["pm2_restart_check.sh", "https_smoke.sh",
             "pm2_restart_check.sh", "mhw_check"]
    r = run(_align_block("apply"))
    assert r.returncode == 0, r.stderr
    steps = _steps(log)
    assert [s.split()[0] for s in steps] == order
    assert "--name tide" in steps[0] and "--prod-port 443" in steps[1]
    assert "--name mhwapi" in steps[2]
    assert (state / "zarr").read_text().strip() == "2.18.7"

    r = run(_align_block("rollback"), "install")  # failed rollback: no restart
    assert r.returncode != 0 and len(_steps(log)) == 4
    r = run(_align_block("rollback"))
    assert r.returncode == 0, r.stderr
    assert [s.split()[0] for s in _steps(log)[4:]] == order
    assert (state / "zarr").read_text().strip() == "2.18.6"


@pytest.mark.parametrize("check", ["traceback-stdout", "traceback-stderr",
                                   "unknown", "empty"])
def test_align_chains_stop_when_pip_check_is_not_recognised(align_chain, check):
    run, log, state = align_chain
    r = run(_align_block("apply"), check=check)     # pip check after install
    assert r.returncode != 0 and _steps(log) == []
    assert (state / "zarr").read_text().strip() == "2.18.7"
    r = run(_align_block("rollback"), check=check)
    assert r.returncode != 0 and _steps(log) == []


@pytest.mark.parametrize("check", ["traceback-stdout", "unknown", "warning"])
def test_align_chain_stops_when_pip_check_is_broken_throughout(
        align_chain, tmp_path, check):
    # Regression: a pip check that fails the same way in prepare and apply
    # was recorded twice, compared equal, and let restart and smoke run.
    run, log, state = align_chain
    d2 = tmp_path / "zarr-align-broken-check"
    r = run('"$ALIGN" prepare --dir "$D"', check=check, dirpath=d2)
    assert r.returncode != 0
    r = run(_align_block("apply"), check=check, dirpath=d2)
    assert r.returncode != 0 and _steps(log) == []
    assert (state / "zarr").read_text().strip() == "2.18.6"   # never installed


# --- Run sheet: preflight A ---------------------------------------------------

def _block_with(needle):
    blocks = [b for b in BLOCKS if needle in b]
    assert len(blocks) == 1, needle
    return blocks[0]


PREFLIGHT_A = _block_with("OK preflight A")
PINNED = {"2.18.7": "ac3dc4033e9ae4e9d7b5e27c97ea3eaf1003cc0a07f010bd83d5134bf8c4b223",
          "2.18.6": "a5334311fae8859870d78561ebb6bd0ab01accc392a6cf4dcaa25d83b4a2f22c"}


def test_preflight_a_pins_the_aligner_hashes():
    script = ALIGN_SCRIPT.read_text()
    assert f"HASH_TO={PINNED['2.18.7']}" in script
    assert f"HASH_FROM={PINNED['2.18.6']}" in script
    for v, h in PINNED.items():
        assert f"{h}  wheels/zarr-{v}-py3-none-any.whl" in PREFLIGHT_A


@pytest.fixture
def preflight_a(align_chain, tmp_path):
    """Preflight A against the fake py311 and the prepared directory of
    align_chain. The fake wheels are empty files, so the pinned hashes are
    swapped for the empty file's."""
    import hashlib
    import shutil
    _, _, state = align_chain
    d = tmp_path / "zarr-align-x"
    empty = hashlib.sha256(b"").hexdigest()
    block = PREFLIGHT_A.replace("/home/odbadmin/.pyenv/versions/py311/bin/python",
                                str(state / "python"))
    for h in PINNED.values():
        block = block.replace(h, empty)
    sha = Path(shutil.which("sha256sum")).parent   # /sbin on macOS

    def run():
        env = {"PATH": f"/usr/bin:/bin:{sha}", "HOME": str(tmp_path),
               "D": str(d), "FAKE_PY": str(state), "FAKE_FAIL": "",
               "TMPDIR": str(tmp_path)}
        return subprocess.run(["bash", "-c", block], env=env, cwd=tmp_path,
                              capture_output=True, text=True, timeout=60)
    return run, d, state


def test_preflight_a_passes_on_the_prepared_state(preflight_a):
    run, d, _ = preflight_a
    r = run()
    assert r.returncode == 0, r.stderr
    assert "OK preflight A" in r.stdout


@pytest.mark.parametrize("change, why", [
    ("numpy", "py311 changed since prepare"),
    ("zarr", "zarr does not import as 2.18.6"),
    ("applied", "already used for apply"),
    ("unprepared", "no prepared marker"),
    ("wheel", "does not match its pinned hash"),
])
def test_preflight_a_refuses_drift(preflight_a, change, why):
    run, d, state = preflight_a
    if change == "numpy":
        (state / "numpy").write_text("2.3.4\n")
    elif change == "zarr":
        (state / "zarr").write_text("2.18.7\n")
    elif change == "applied":
        (d / "applied").write_text("")
    elif change == "unprepared":
        (d / "prepared").unlink()
    else:
        (d / "wheels" / "zarr-2.18.7-py3-none-any.whl").write_text("x")
    r = run()
    assert r.returncode != 0
    assert why in r.stderr and "do not apply" in r.stderr
    assert "OK preflight A" not in r.stdout


# --- Run sheet: MHW checks ----------------------------------------------------

MHW_BLOCK = _block_with("mhw_check() {")
PY311 = "/home/odbadmin/.pyenv/versions/py311/bin"
POINT = (b'[{"lon":135.125,"lat":15.125,"date":"2024-01-01","level":1},'
         b'{"lon":135.125,"lat":15.125,"date":"2024-02-01","level":0},'
         b'{"lon":135.125,"lat":15.125,"date":"2024-03-01","level":0}]')
RESPONSES = {
    "/api/swagger/mhw/openapi.json": (
        "200 application/json",
        b'{"openapi":"3.1.0","paths":{"/api/mhw":{},"/api/mhw/csv":{}}}'),
    "/api/swagger/mhw": (
        "200 text/html; charset=utf-8",
        b"<html>url: '/api/swagger/mhw/openapi.json'</html>"),
    "/api/mhw?lon0=135&lat0=15&start=2024-01-01&end=2024-03-01": (
        "200 application/json", POINT),
    "/api/mhw": (
        "422 application/json",
        b'{"detail":[{"loc":["query","lon0"]},{"loc":["query","lat0"]}]}'),
}
FAKE_CURL = r'''#!/bin/bash
# Fake curl: -o FILE, -w ignored (prints "STATUS TYPE"), URL last.
while [ $# -gt 1 ]; do
  case "$1" in -o) out=$2; shift 2 ;; -w|--proto|--max-time) shift 2 ;; *) shift ;; esac
done
url=$1
echo "$url" >> "$FAKE_WEB/requests"
case "$url" in https://eco.odb.ntu.edu.tw/*) ;; *) exit 6 ;; esac
key=$(printf '%s' "${url#https://eco.odb.ntu.edu.tw}" | shasum | cut -c1-12)
cp "$FAKE_WEB/$key.body" "$out"
cat "$FAKE_WEB/$key.status"
'''
FAKE_SS = '#!/bin/bash\ncat "$FAKE_WEB/ss"\n'
FAKE_PS = r'''#!/bin/bash
# Fake ps: -o ppid= -p PID or -o args= -p PID, from "PID PPID ARGS" lines.
field=$2; pid=$4
line=$(grep "^$pid " "$FAKE_WEB/ps") || exit 1
case "$field" in
  ppid=) echo "$line" | cut -d' ' -f2 ;;
  args=) echo "$line" | cut -d' ' -f3- ;;
esac
'''


@pytest.fixture
def mhw(tmp_path):
    import hashlib
    import shutil
    import sys
    web = tmp_path / "web"
    web.mkdir()
    for path, (status, body) in RESPONSES.items():
        key = hashlib.sha1(path.encode()).hexdigest()[:12]
        (web / f"{key}.status").write_text(status)
        (web / f"{key}.body").write_bytes(body)
    (web / "ss").write_text(
        'LISTEN 0 2048 127.0.0.1:8030 0.0.0.0:* users:(("gunicorn",pid=12,fd=6),'
        '("gunicorn",pid=11,fd=6),("gunicorn",pid=10,fd=6))\n')
    gunicorn = f"{PY311}/python3.11 {PY311}/gunicorn mhw_app:app -w 2 --reload"
    (web / "ps").write_text(
        f"9 1 bash {MHW_LAUNCHER}\n10 9 {gunicorn}\n"
        f"11 10 {gunicorn}\n12 10 {gunicorn}\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, text in (("curl", FAKE_CURL), ("ss", FAKE_SS), ("ps", FAKE_PS)):
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)
    (bin_ / "python3").symlink_to(sys.executable)
    (bin_ / "shasum").symlink_to(shutil.which("shasum"))

    def run(call):
        env = {"PATH": f"{bin_}:/usr/bin:/bin", "HOME": str(tmp_path),
               "FAKE_WEB": str(web)}
        return subprocess.run(["bash", "-c", MHW_BLOCK + "\n" + call], env=env,
                              cwd=tmp_path, capture_output=True, text=True,
                              timeout=60)
    return run, web, tmp_path


def test_mhw_check_passes_and_compares_with_the_baseline(mhw):
    run, web, tmp = mhw
    r = run(f'mhw_check "{tmp}/before"')
    assert r.returncode == 0, r.stderr
    assert "OK mhw process: master 10 " in r.stdout
    assert (tmp / "before" / "point.json").read_bytes() == POINT
    r = run(f'mhw_check "{tmp}/after" "{tmp}/before"')
    assert r.returncode == 0, r.stderr
    assert "point query identical" in r.stdout
    r = run(f'mhw_check "{tmp}/after" "{tmp}/before"')   # never overwrites
    assert r.returncode != 0 and "PASS" not in r.stdout


def test_mhw_check_only_sends_reads_to_the_public_frontend(mhw):
    run, web, tmp = mhw
    assert run(f'mhw_check "{tmp}/out"').returncode == 0
    sent = (web / "requests").read_text().splitlines()
    assert sent == ["https://eco.odb.ntu.edu.tw" + p for p in RESPONSES]
    for c in _commands(MHW_BLOCK):
        if re.search(r"\bcurl\b", c):
            assert not re.search(r"\s(-X|--request|-d|--data\S*|-F|--form|-T|-L)\s", c), c
    assert "/api/mhw/csv?" not in MHW_BLOCK and "/api/mhw/plot" not in MHW_BLOCK
    code = [c for c in _commands(MHW_BLOCK) if not c.startswith("#")]
    assert not [c for c in code if "pm2" in c]


def _rewrite(web, path, status=None, body=None):
    import hashlib
    key = hashlib.sha1(path.encode()).hexdigest()[:12]
    if status is not None:
        (web / f"{key}.status").write_text(status)
    if body is not None:
        (web / f"{key}.body").write_bytes(body)


POINT_PATH = "/api/mhw?lon0=135&lat0=15&start=2024-01-01&end=2024-03-01"


@pytest.mark.parametrize("breakage, why", [
    (lambda w: _rewrite(w, POINT_PATH, status="500 application/json"), "gave '500"),
    (lambda w: _rewrite(w, "/api/mhw", status="200 application/json"), "gave '200"),
    (lambda w: _rewrite(w, "/api/swagger/mhw", status="200 text/plain"), "gave '200"),
    (lambda w: _rewrite(w, POINT_PATH, body=b"[]"), "lacks an expected feature"),
    (lambda w: _rewrite(w, "/api/swagger/mhw/openapi.json",
                        body=b'{"paths":{"/api/mhw":{}}}'), "lacks an expected feature"),
])
def test_mhw_http_check_fails_on_unexpected_responses(mhw, breakage, why):
    run, web, tmp = mhw
    breakage(web)
    r = run(f'mhw_check "{tmp}/out"')
    assert r.returncode != 0 and why in r.stderr, r.stderr
    assert "PASS" not in r.stdout


def test_mhw_http_check_fails_when_the_point_query_changes(mhw):
    run, web, tmp = mhw
    assert run(f'mhw_check "{tmp}/before"').returncode == 0
    _rewrite(web, POINT_PATH, body=POINT.replace(b'"level":1', b'"level":2'))
    r = run(f'mhw_check "{tmp}/after" "{tmp}/before"')
    assert r.returncode != 0 and "differs from" in r.stderr
    assert "PASS" not in r.stdout


@pytest.mark.parametrize("ps, ss, why", [
    # Gunicorn from another interpreter under the launcher.
    (f"9 1 bash {MHW_LAUNCHER}\n10 9 /usr/bin/python3 /usr/bin/gunicorn mhw_app:app\n",
     None, "not the py311 Gunicorn"),
    # The listener's parent is not the MHW launcher.
    (f"9 1 bash /tmp/other.sh\n10 9 {PY311}/python3.11 {PY311}/gunicorn mhw_app:app\n",
     None, "no 8030 listener under the MHW launcher"),
    # Nothing listens on 8030.
    (None, "", "no 8030 listener under the MHW launcher"),
    # Two listening masters under the launcher: ambiguous, refuse.
    (f"9 1 bash {MHW_LAUNCHER}\n10 9 {PY311}/python3.11 {PY311}/gunicorn mhw_app:app\n"
     f"13 9 {PY311}/python3.11 {PY311}/gunicorn mhw_app:app\n",
     'users:(("gunicorn",pid=10,fd=6),("gunicorn",pid=13,fd=6))\n',
     "two 8030 masters"),
])
def test_mhw_proc_check_fails_on_the_wrong_process(mhw, ps, ss, why):
    run, web, tmp = mhw
    if ps is not None:
        (web / "ps").write_text(ps)
    if ss is not None:
        (web / "ss").write_text(ss)
    r = run(f'mhw_check "{tmp}/out"')
    assert r.returncode != 0 and why in r.stderr, r.stderr
    assert not (tmp / "out").exists()     # no GET after a failed process check


def test_service_order_stays_a_proposal_until_approved():
    section = TEXT[TEXT.index("### Service handling during the install"):
                   TEXT.index("### MHW restart and post-restart smoke")]
    flat = " ".join(section.split())
    assert "Proposed order, not validated" in flat
    assert "This order is **not approved**" in flat
    assert "never against the real `mhwapi`" in flat
    for overclaim in ("(recommended)", "Recommendation:", "race is bounded",
                      "Validated tools only", "restart failed workers"):
        assert overclaim not in flat, overclaim
    sheet = " ".join(TEXT[TEXT.index("### Production run sheet"):].split())
    assert "Do not run `apply` while any item is open." in sheet
    assert "- [ ] Owner: approve the service order" in sheet
