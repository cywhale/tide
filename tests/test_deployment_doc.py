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
