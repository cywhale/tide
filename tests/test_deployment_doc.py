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
