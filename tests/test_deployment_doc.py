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
# py311_zarr_state.sh stand-in: the fake py311's zarr version decides the
# state; ZSTATE_UNKNOWN forces STATE unknown.
FAKE_ZSTATE = r'''#!/bin/bash
echo "zstate $*" >> "$STEPS_LOG"
expect=""
while [ $# -gt 0 ]; do case "$1" in --expect) expect=$2; shift 2 ;; *) shift ;; esac; done
[ -z "$ZSTATE_UNKNOWN" ] || { echo "STATE unknown"; exit 1; }
case "$(cat "$FAKE_PY/zarr")" in 2.18.6) s=before ;; 2.18.7) s=after ;; *) s=unknown ;; esac
echo "STATE $s"
[ "$s" != unknown ] && { [ -z "$expect" ] || [ "$s" = "$expect" ]; }
'''


def _align_block(mode):
    blocks = [b for b in BLOCKS if f'"$ALIGN" {mode} --dir "$D"' in b]
    assert len(blocks) == 1, mode
    return blocks[0]


MHW_LAUNCHER = "/home/odbadmin/proj/marineheatwave/API/conf/start_app.sh"


def _chain(block):
    cmds = _commands(block)
    assert len(cmds) == 1, cmds                # one && chain, nothing after it
    return [" ".join(s.split()) for s in cmds[0].split("&&")]


def test_align_blocks_require_a_known_state_before_anything_starts():
    assert len([b for b in BLOCKS if '"$ALIGN" prepare --dir "$D"' in b]) == 1
    assert _chain(_align_block("apply")) == [
        '"$ALIGN" apply --dir "$D"', '"$ZSTATE" --dir "$D" --expect after']
    assert _chain(_align_block("rollback")) == [
        "stop_services", '"$ALIGN" rollback --dir "$D"',
        '"$ZSTATE" --dir "$D" --expect before',
        'start_services "$D-rollback-smoke" "$D-mhw-rollback"']
    # The normal start is its own step, after the apply block succeeded.
    assert 'start_services "$D-smoke" "$D-mhw-after"' in ALL_COMMANDS
    assert "stop_services" in ALL_COMMANDS
    zarr = TEXT[TEXT.index("## Py311 Zarr alignment"):TEXT.index("## Operational Notes")]
    cmds = [c for b in re.findall(r"^ *```bash\n(.*?)^ *```", zarr, flags=re.S | re.M)
            for c in _commands(textwrap.dedent(b)) if not c.startswith("#")]
    assert cmds
    assert not any("--update-env" in c or re.search(r"\bpm2 (start|stop|restart)\b", c)
                   for c in cmds)


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
    zstate = tmp_path / "zstate"
    zstate.write_text(FAKE_ZSTATE)
    zstate.chmod(0o755)

    # The run sheet defines mhw_check and the service helpers in earlier
    # blocks of the same shell; here they only record that they ran.
    mhw_stub = ('mhw_check() { echo "mhw_check $*" >> "$STEPS_LOG"; }\n'
                'stop_services() { echo "stop_services" >> "$STEPS_LOG"; }\n'
                'start_services() { echo "start_services $*" >> "$STEPS_LOG"; }\n')

    def run(block, fail="", check="conflicts", dirpath=d, zstate_unknown=""):
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "SRC": str(src),
               "ALIGN": str(wrapper), "D": str(dirpath), "FAKE_PY": str(state),
               "FAKE_FAIL": fail, "FAKE_CHECK": check, "STEPS_LOG": str(log),
               "ZSTATE": str(zstate), "ZSTATE_UNKNOWN": zstate_unknown}
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


def test_align_apply_then_rollback_chains(align_chain):
    run, log, state = align_chain
    r = run(_align_block("apply"))
    assert r.returncode == 0, r.stderr
    assert _steps(log) == ["zstate --dir {} --expect after".format(log.parent / "zarr-align-x")]
    assert (state / "zarr").read_text().strip() == "2.18.7"

    log.unlink()
    r = run(_align_block("rollback"), "install")  # failed rollback: nothing starts
    assert r.returncode != 0 and _steps(log) == ["stop_services"]
    log.unlink()
    r = run(_align_block("rollback"))
    assert r.returncode == 0, r.stderr
    steps = _steps(log)
    assert [s.split()[0] for s in steps] == ["stop_services", "zstate", "start_services"]
    assert steps[1].endswith("--expect before")
    assert (state / "zarr").read_text().strip() == "2.18.6"


def test_rollback_chain_starts_nothing_when_the_state_is_unknown(align_chain):
    run, log, _ = align_chain
    assert run(_align_block("apply")).returncode == 0
    log.unlink()
    r = run(_align_block("rollback"), zstate_unknown="1")
    assert r.returncode != 0
    assert [s.split()[0] for s in _steps(log)] == ["stop_services", "zstate"]


@pytest.mark.parametrize("check", ["traceback-stdout", "traceback-stderr",
                                   "unknown", "empty"])
def test_align_chains_stop_when_pip_check_is_not_recognised(align_chain, check):
    run, log, state = align_chain
    r = run(_align_block("apply"), check=check)     # pip check after install
    assert r.returncode != 0 and _steps(log) == []
    assert (state / "zarr").read_text().strip() == "2.18.7"
    r = run(_align_block("rollback"), check=check)
    assert r.returncode != 0 and _steps(log) == ["stop_services"]   # nothing starts


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
    # Also inside an && list, where bash ignores set -e in the function.
    r = run(f'true && mhw_check "{tmp}/after" "{tmp}/before" && echo NEXT')
    assert r.returncode != 0 and "already exists" in r.stderr
    assert "PASS" not in r.stdout and "NEXT" not in r.stdout


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
    section = TEXT[TEXT.index("### Dask and Zarr evidence"):
                   TEXT.index("### MHW post-start smoke")]
    flat = " ".join(section.split())
    assert "### Service handling: stop first (proposed, not approved)" in section
    assert "it has never run in production" in flat
    assert "Decision: stop the Dask scheduler too." in flat
    assert "Stopping the APIs does not remove the import risk." in flat
    assert "It shows one moment, not what will arrive during a later window." in flat
    for heading in ("Tested:", "Derived from source, not run against the real services:",
                    "Unknown until the window:"):
        assert heading in flat, heading
    for overclaim in ("production-validated", "validated in production",
                      "(recommended)", "Recommendation:", "race is bounded",
                      "No running process imports Zarr", "identical 2.18.6 code",
                      "Zarr code", "pre_stop kills"):
        assert overclaim not in flat, overclaim
    sheet = " ".join(TEXT[TEXT.index("### Production run sheet"):
                          TEXT.index("## Operational Notes")].split())
    assert "Do not run `apply` while any item is open." in sheet
    for item in ("- [ ] Owner: approve the stop-first order",
                 "- [ ] MHW maintainer: agree to stopping and starting",
                 "- [ ] Owner and MHW maintainer: the window",
                 "- [ ] Owner: reachable during the whole window"):
        assert item in sheet, item
    assert "Services start only against a known package state" in sheet
    assert "With `STATE unknown`, nothing starts." in sheet


def test_no_runbook_text_claims_pm2_runs_pre_stop():
    flat = " ".join(TEXT.split())
    assert "pre_stop` kills every" not in flat
    assert "pre_stop of `tide` kills" not in flat
    assert "PM2 5.4.2 never runs" in flat


# --- Run sheet: preflight B ---------------------------------------------------

PREFLIGHT_B = _block_with("OK preflight B")


@pytest.fixture
def preflight_b(tmp_path):
    """Preflight B with the production paths moved into tmp_path and every
    external command replaced by a stub driven by files in tmp_path/w."""
    w = tmp_path / "w"
    w.mkdir()
    pm2h = tmp_path / "pm2home"
    pm2h.mkdir()
    (pm2h / "pm2.pid").write_text("3352\n")
    tls = tmp_path / "tls"
    for f in ("certs/ca.crt", "certs/mhw-client.crt", "private/mhw-client.key"):
        (tls / f).parent.mkdir(parents=True, exist_ok=True)
        (tls / f).write_text("")
    src = tmp_path / "src"
    (src / "conf").mkdir(parents=True)
    stubs = {
        "pm2_restart_check.sh": '#!/bin/bash\necho "daemon ok"\n',
        "pm2_app_lifecycle.sh": ('#!/bin/bash\nmode=$1; shift\n'
                                 'while [ $# -gt 0 ]; do case "$1" in --name) n=$2; shift 2 ;; *) shift ;; esac; done\n'
                                 'if [ "$mode" = pid ]; then echo 101867; exit 0; fi\n'
                                 'echo "$(cat "$W/status.$n" 2>/dev/null || echo online) 1 0"\n'),
        "dask_scheduler_check.sh": '#!/bin/bash\necho "$*" > "$W/dask.args"\n[ ! -e "$W/dask.fail" ]\n',
    }
    for name, body in stubs.items():
        (src / "conf" / name).write_text(body)
        (src / "conf" / name).chmod(0o755)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "ps").write_text(
        '#!/bin/bash\nwhile [ $# -gt 0 ]; do case "$1" in -p) p=$2; shift 2 ;; *) shift ;; esac; done\n'
        'cat "$W/ps.$p"\n')
    (bindir / "pgrep").write_text('#!/bin/bash\n[ -e "$W/pgrep.hit" ]\n')
    for f in ("ps", "pgrep"):
        (bindir / f).chmod(0o755)
    (w / "ps.3352").write_text(f"PM2 v5.4.2: God Daemon ({pm2h})\n")
    (w / "ps.96946").write_text("bash /home/odbadmin/proj/marineheatwave/API/conf/start_app.sh\n")
    block = (PREFLIGHT_B.replace("/home/odbadmin/.pm2", str(pm2h))
             .replace("/home/odbadmin/.config/dask/tls", str(tls)))

    def run():
        env = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path),
               "SRC": str(src), "W": str(w)}
        return subprocess.run(["bash", "-c", block], env=env, cwd=tmp_path,
                              capture_output=True, text=True, timeout=60)
    return run, w


def test_preflight_b_passes_and_asks_dask_for_the_nanny_and_clients(preflight_b):
    run, w = preflight_b
    r = run()
    assert r.returncode == 0, r.stderr
    assert "OK preflight B" in r.stdout
    assert (w / "dask.args").read_text().split() == [
        "ready", "--workers", "1", "--nanny-pid", "101867", "--clients", "4",
        "--timeout", "10"]


@pytest.mark.parametrize("change, why", [
    ("version", "check pre_stop support"),
    ("launcher", "no longer PID 96946"),
    ("update", "the MHW update is running"),
    ("stopped", "dask-worker is not online"),
    ("dask", "Dask is not one worker"),
])
def test_preflight_b_refuses(preflight_b, change, why):
    run, w = preflight_b
    if change == "version":
        (w / "ps.3352").write_text((w / "ps.3352").read_text().replace("5.4.2", "6.0.1"))
    elif change == "launcher":
        (w / "ps.96946").write_text("sleep 60\n")
    elif change == "update":
        (w / "pgrep.hit").write_text("")
    elif change == "stopped":
        (w / "status.dask-worker").write_text("stopped")
    else:
        (w / "dask.fail").write_text("")
    r = run()
    assert r.returncode != 0
    assert why in r.stderr and "do not apply" in r.stderr
    assert "OK preflight B" not in r.stdout


# --- Run sheet: maintenance helpers ------------------------------------------

HELPERS = _block_with("start_services() (")

# pm2_app_lifecycle.sh stand-in: one status file per app; stop and start flip
# it, FAIL_AT ("start tide", "stop dask-worker", ...) fails that one call.
FAKE_LIFE = r'''#!/bin/bash
mode=$1; shift
n=""; match=""; exe=""
while [ $# -gt 0 ]; do
  case "$1" in --name) n=$2; shift 2 ;; --match) match=$2; shift 2 ;;
    --expect-exec) exe=$2; shift 2 ;; *) shift ;; esac
done
f="$W/status.$n"
case "$mode" in
  status) echo "$(cat "$f") 1 0" ;;
  pid) echo 101867 ;;
  stop|start)
    echo "$mode $n match=$match exe=$exe" >> "$STEPS_LOG"
    [ "${FAIL_AT:-}" != "$mode $n" ] || exit 1
    if [ "$mode" = stop ]; then echo stopped > "$f"; else echo online > "$f"; fi ;;
esac
'''
FAKE_STEP = '#!/bin/bash\necho "$(basename "$0") $*" >> "$STEPS_LOG"\n[ "${FAIL_AT:-}" != "$(basename "$0") $1" ]\n'


@pytest.fixture
def helpers(tmp_path):
    w = tmp_path / "w"
    w.mkdir()
    src = tmp_path / "src"
    (src / "conf").mkdir(parents=True)
    for name, body in (("pm2_app_lifecycle.sh", FAKE_LIFE), ("https_smoke.sh", FAKE_STEP),
                       ("dask_scheduler_check.sh", FAKE_STEP)):
        (src / "conf" / name).write_text(body)
        (src / "conf" / name).chmod(0o755)
    proc = tmp_path / "proc"
    proc.mkdir()
    log = tmp_path / "steps.log"
    block = HELPERS.replace("proc=/proc", f"proc={proc}")
    assert block != HELPERS
    mhw_stub = 'mhw_check() { echo "mhw_check $*" >> "$STEPS_LOG"; [ "${FAIL_AT:-}" != mhw_check ]; }\n'

    def run(cmd, status=None, fail=""):
        for a in ("dask-scheduler", "dask-worker", "tide", "mhwapi"):
            (w / f"status.{a}").write_text((status or {}).get(a, "online"))
        log.unlink(missing_ok=True)
        env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "SRC": str(src),
               "DASKC": str(src / "conf" / "dask_scheduler_check.sh"),
               "D": str(tmp_path / "d"), "W": str(w), "STEPS_LOG": str(log),
               "FAIL_AT": fail}
        r = subprocess.run(["bash", "-c", mhw_stub + block + "\n" + cmd], env=env,
                           cwd=tmp_path, capture_output=True, text=True, timeout=60)
        return r, [" ".join(l.split()[:2]) if not l.startswith("dask_scheduler_check.sh")
                   else l for l in (log.read_text().splitlines() if log.exists() else [])]
    return run, proc


def test_app_passes_the_match_word_and_executable_per_app(helpers):
    run, _ = helpers
    r, _ = run("app stop mhwapi && app stop tide && app stop dask-worker && app stop dask-scheduler")
    assert r.returncode == 0, r.stderr
    log = (helpers[1].parent / "steps.log").read_text().splitlines()
    py = "/home/odbadmin/.pyenv/versions/py311/bin"
    assert log == [
        f"stop mhwapi match=mhw_app:app exe={MHW_LAUNCHER}",
        f"stop tide match=tide_app:app exe={py}/gunicorn",
        f"stop dask-worker match={py}/dask-worker exe={py}/dask-worker",
        f"stop dask-scheduler match={py}/dask-scheduler exe={py}/dask-scheduler",
    ]
    r, _ = run("app stop gateway")
    assert r.returncode != 0 and "unknown app" in r.stderr


def test_stop_services_order_and_quiescence(helpers):
    run, _ = helpers
    r, steps = run("stop_services")
    assert r.returncode == 0, r.stderr
    assert steps == ["stop mhwapi", "stop tide", "dask_scheduler_check.sh drained",
                     "stop dask-worker", "stop dask-scheduler"]
    assert "OK quiescent" in r.stdout and "OK services stopped" in r.stdout


@pytest.mark.parametrize("fail, done", [
    ("stop tide", ["stop mhwapi", "stop tide"]),
    ("dask_scheduler_check.sh drained",
     ["stop mhwapi", "stop tide", "dask_scheduler_check.sh drained"]),
    ("stop dask-worker", ["stop mhwapi", "stop tide", "dask_scheduler_check.sh drained",
                          "stop dask-worker"]),
])
def test_stop_services_stops_at_the_first_failure(helpers, fail, done):
    run, _ = helpers
    r, steps = run("stop_services && echo APPLY_WOULD_RUN", fail=fail)
    assert r.returncode != 0
    assert steps == done
    assert "APPLY_WOULD_RUN" not in r.stdout


def test_stop_services_skips_stopped_apps_and_the_drain_without_a_scheduler(helpers):
    run, _ = helpers
    r, steps = run("stop_services", status={"mhwapi": "stopped", "dask-scheduler": "stopped"})
    assert r.returncode == 0, r.stderr
    assert steps == ["stop tide", "stop dask-worker"]
    assert "note: mhwapi already stopped" in r.stdout


@pytest.mark.parametrize("where", ["cmdline", "maps"])
def test_py311_quiescent_finds_a_process_using_py311(helpers, where):
    run, proc = helpers
    (proc / "4242").mkdir()
    py = "/home/odbadmin/.pyenv/versions/py311/"
    (proc / "4242" / "cmdline").write_bytes(
        (py + "bin/python3.11\0x.py\0").encode() if where == "cmdline" else b"python\0x.py\0")
    (proc / "4242" / "maps").write_text(
        f"7f00 r-xp 0 00:00 0 {py}lib/python3.11/site-packages/numpy/core/_m.so\n"
        if where == "maps" else "7f00 r-xp 0 00:00 0 /usr/lib/libc.so\n")
    r, steps = run("stop_services && echo APPLY_WOULD_RUN")
    assert r.returncode != 0
    assert "processes still use py311" in r.stderr and "4242" in r.stderr
    assert "APPLY_WOULD_RUN" not in r.stdout


def test_py311_quiescent_counts_unreadable_maps(helpers):
    run, proc = helpers
    (proc / "77").mkdir()
    (proc / "77" / "cmdline").write_bytes(b"/usr/sbin/sshd\0")
    (proc / "77" / "maps").write_text("")
    (proc / "77" / "maps").chmod(0)
    try:
        r, _ = run("py311_quiescent")
    finally:
        (proc / "77" / "maps").chmod(0o644)
    assert r.returncode == 0, r.stderr
    assert "(1 processes of other users not readable)" in r.stdout


def test_start_services_order(helpers):
    run, _ = helpers
    stopped = {a: "stopped" for a in ("dask-scheduler", "dask-worker", "tide", "mhwapi")}
    r, steps = run('start_services "$D-smoke" "$D-mhw-after"', status=stopped)
    assert r.returncode == 0, r.stderr
    d = str(helpers[1].parent / "d")
    assert steps == [
        "start dask-scheduler", "dask_scheduler_check.sh ready --workers 0",
        "start dask-worker", "dask_scheduler_check.sh ready --workers 1 --nanny-pid 101867",
        "start tide", "https_smoke.sh --host",
        "start mhwapi", f"mhw_check {d}-mhw-after",
        "dask_scheduler_check.sh ready --workers 1 --nanny-pid 101867 --clients 4"]
    assert "OK services started" in r.stdout


@pytest.mark.parametrize("fail, last", [
    ("start dask-scheduler", "start dask-scheduler"),
    ("start dask-worker", "start dask-worker"),
    ("dask_scheduler_check.sh ready", "dask_scheduler_check.sh ready --workers 0"),
    ("https_smoke.sh --host", "https_smoke.sh --host"),
    ("mhw_check", None),
])
def test_start_services_stops_at_the_first_failure(helpers, fail, last):
    run, _ = helpers
    stopped = {a: "stopped" for a in ("dask-scheduler", "dask-worker", "tide", "mhwapi")}
    # As in the run sheet's recovery chains: inside an && list.
    r, steps = run('true && start_services "$D-smoke" "$D-mhw-after" && echo NEXT',
                   status=stopped, fail=fail)
    assert r.returncode != 0
    assert "OK services started" not in r.stdout and "NEXT" not in r.stdout
    if last:
        assert steps[-1] == last
    else:
        assert steps[-1].startswith("mhw_check")
    if fail in ("start dask-scheduler", "start dask-worker", "dask_scheduler_check.sh ready"):
        assert "start tide" not in steps and "start mhwapi" not in steps


def test_start_services_starts_only_what_is_down(helpers):
    run, _ = helpers
    r, steps = run('start_services "$D-s" "$D-m"', status={"tide": "errored", "mhwapi": "stopped"})
    assert r.returncode == 0, r.stderr
    assert [s for s in steps if s.startswith("start")] == ["start tide", "start mhwapi"]
    assert steps[0] == "dask_scheduler_check.sh ready --workers 1"
    assert "note: dask-scheduler already online" in r.stdout


def test_start_services_refuses_an_app_in_an_unexpected_state(helpers):
    run, _ = helpers
    r, steps = run('start_services "$D-s" "$D-m"', status={"dask-scheduler": "launching"})
    assert r.returncode != 0 and "unexpected state" in r.stderr
    assert steps == []


def test_recovery_blocks_gate_starts_on_a_known_state():
    r0 = [b for b in BLOCKS if 'start_services "$D-r0-smoke"' in b]
    assert len(r0) == 1
    assert _chain(r0[0]) == ['"$ZSTATE" --dir "$D" --expect before',
                             'start_services "$D-r0-smoke" "$D-mhw-r0"']
    r2 = " ".join(TEXT.split())
    assert '`"$ZSTATE" --dir "$D" --expect after && start_services "$D-r2-smoke" "$D-mhw-r2"`' in r2
