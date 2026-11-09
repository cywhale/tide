"""Production launcher (`conf/start_app.sh`) behavior.

The script runs for real in a sandbox: the VM24 py311 Gunicorn path is
rewritten to a temp path, and stubs record invocations. It must run only an
explicit absolute GUNICORN_BIN or the py311 Gunicorn, never `uv run` or a PATH
gunicorn. It must not require or pass Dask client credentials to Tide.
"""
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
START_APP = REPO_ROOT / "conf" / "start_app.sh"
VM24_PY311_GUNICORN = "/home/odbadmin/.pyenv/versions/py311/bin/gunicorn"

STUB = ('#!/bin/bash\n'
        'echo "$0|${DASK_SCHEDULER_ADDRESS-}|${DASK_TLS_CA_FILE-}|'
        '${DASK_TLS_CLIENT_CERT-}|${DASK_TLS_CLIENT_KEY-}|$*" >> "$STUB_LOG"\n')


def _stub(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STUB)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def sandbox(tmp_path):
    text = START_APP.read_text()
    assert VM24_PY311_GUNICORN in text
    assert "DASK_TLS_DIR=" not in text
    assert "/home/odbadmin/.config/dask/tls" not in text
    py311 = tmp_path / "py311" / "gunicorn"
    script = tmp_path / "start_app.sh"
    script.write_text(text.replace(VM24_PY311_GUNICORN, str(py311)))
    pathbin = tmp_path / "pathbin"
    _stub(pathbin / "uv")
    _stub(pathbin / "gunicorn")
    return {"root": tmp_path, "script": script, "py311": py311,
            "pathbin": pathbin, "log": tmp_path / "stub.log"}


def _run(sb, **env):
    full = {"PATH": f"{sb['pathbin']}:/usr/bin:/bin", "STUB_LOG": str(sb["log"])}
    full.update(env)
    r = subprocess.run(["bash", str(sb["script"])], cwd=sb["root"], env=full,
                       capture_output=True, text=True, timeout=30)
    calls = sb["log"].read_text().splitlines() if sb["log"].exists() else []
    return r, calls


def test_default_runs_py311_gunicorn_without_dask_client_env(sandbox):
    _stub(sandbox["py311"])
    r, calls = _run(sandbox)
    assert r.returncode == 0, r.stderr
    (call,) = calls
    exe, addr, ca, cert, key, args = call.split("|")
    assert exe == str(sandbox["py311"])
    assert (addr, ca, cert, key) == ("", "", "", "")
    assert args.startswith("tide_app:app ")
    assert "-b 127.0.0.1:8040" in args and "--reload" in args


def test_inherited_dask_client_credentials_are_not_passed_to_tide(sandbox):
    _stub(sandbox["py311"])
    r, calls = _run(
        sandbox,
        DASK_SCHEDULER_ADDRESS="tls://127.0.0.1:8786",
        DASK_TLS_CA_FILE="/secret/ca.crt",
        DASK_TLS_CLIENT_CERT="/secret/tide.crt",
        DASK_TLS_CLIENT_KEY="/secret/tide.key",
    )
    assert r.returncode == 0, r.stderr
    call = calls[0].split("|")
    assert call[1:5] == ["", "", "", ""]


def test_explicit_absolute_gunicorn_bin_wins(sandbox):
    _stub(sandbox["py311"])
    other = _stub(sandbox["root"] / "other" / "gunicorn")
    r, calls = _run(sandbox, GUNICORN_BIN=str(other))
    assert r.returncode == 0, r.stderr
    assert [c.split("|")[0] for c in calls] == [str(other)]


def test_no_py311_never_falls_back_to_uv_or_path(sandbox):
    # py311 stub absent; `uv` and `gunicorn` stubs are on PATH and must not run.
    r, calls = _run(sandbox)
    assert r.returncode == 127
    assert calls == []
    assert "py311 Gunicorn not found" in r.stderr


@pytest.mark.parametrize("value", ["gunicorn", "uv", "bin/gunicorn"])
def test_relative_gunicorn_bin_is_refused(sandbox, value):
    _stub(sandbox["py311"])
    r, calls = _run(sandbox, GUNICORN_BIN=value)
    assert r.returncode == 127
    assert calls == []
    assert "absolute path" in r.stderr


def test_non_executable_gunicorn_bin_is_refused(sandbox):
    _stub(sandbox["py311"])
    r, calls = _run(sandbox, GUNICORN_BIN=str(sandbox["root"] / "missing" / "gunicorn"))
    assert r.returncode == 127
    assert calls == []
    assert "not executable" in r.stderr


def test_start_app_sh_syntax():
    subprocess.run(["bash", "-n", str(START_APP)], check=True)
