"""v0.3.2 operator scripts: atomic manifest replacement and HTTPS smoke.

Both scripts run for real against sandboxes (a temp release directory and a
temp git checkout; a `curl` stub on PATH), so the fail-fast control flow is
exercised, not just the syntax.
"""
import os
import stat
import subprocess
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


def _replace(sb, mode, path_prefix=None, **over):
    args = {"rel": sb["rel"], "work": sb["work"]}
    if mode == "deploy":
        args.update(src=sb["src"], commit=sb["commit"])
    args.update(over)
    cmd = ["bash", str(REPLACE), mode]
    for k, v in args.items():
        cmd += [f"--{k}", str(v)]
    env = dict(os.environ)
    if path_prefix:
        env["PATH"] = f"{path_prefix}:{env['PATH']}"
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)


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

    def run(mode, **resp):
        env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", CURL_LOG=str(log))
        for port, (code, body) in resp.items():
            env[f"FAKE_{port[1:]}_CODE"] = str(code)
            env[f"FAKE_{port[1:]}_BODY"] = body
        out = tmp_path / f"out-{len(list(tmp_path.glob('out-*')))}"
        r = subprocess.run(["bash", str(SMOKE), "--host", "eco.odb.ntu.edu.tw",
                            "--out", str(out), "--mode", mode],
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


def test_ops_scripts_syntax():
    for script in (REPLACE, SMOKE):
        subprocess.run(["bash", "-n", str(script)], check=True)
