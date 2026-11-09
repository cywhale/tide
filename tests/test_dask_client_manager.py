"""The old Tide Dask manager import remains a no-op compatibility shim."""
import ast
import subprocess
import sys
from pathlib import Path

import src.dask_client_manager as DCM

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compatibility_functions_never_create_or_close_a_client():
    assert DCM.get_dask_client("tideapi") is None
    assert DCM.close_dask_client("tideapi") is None


def test_tide_app_imports_without_dask_installed():
    code = r'''import builtins
import importlib.util

real_find_spec = importlib.util.find_spec
real_import = builtins.__import__

def find_spec(name, *args, **kwargs):
    if name == "dask" or name.startswith("dask.") or \
            name == "distributed" or name.startswith("distributed."):
        return None
    return real_find_spec(name, *args, **kwargs)

def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in {"dask", "distributed"}:
        raise ModuleNotFoundError("Dask import blocked by test")
    return real_import(name, *args, **kwargs)

importlib.util.find_spec = find_spec
builtins.__import__ = guarded_import
import tide_app
assert tide_app.app is not None
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_compatibility_shim_has_no_dask_imports():
    source = (REPO_ROOT / "src" / "dask_client_manager.py").read_text()
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any(
        (node.module or "").split(".")[0] in {"dask", "distributed"}
        or any(alias.name.split(".")[0] in {"dask", "distributed"}
               for alias in getattr(node, "names", []))
        for node in imports
    )
