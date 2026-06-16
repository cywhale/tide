"""Stage 0 G0 + Stage 3 runtime-migration smoke tests.

Verifies the app imports, pinned runtime versions, and that the migrated
runtime serves BOTH stores through the adapter (D9): the tpxo10 canonical
(default) and the legacy tpxo9 store selected by TIDE_ZARR_PATH (the
rollback path).
"""
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TPXO9_STORE = REPO_ROOT / "data" / "tpxo9.zarr"
TPXO10_STORE = REPO_ROOT / "data" / "tpxo10.zarr"


def test_app_importable():
    import tide_app

    assert tide_app.app is not None


def test_pinned_runtime_versions():
    import pyTMD
    import zarr

    assert pyTMD.version.full_version == "v2.2.8"
    assert zarr.__version__ == "2.18.7"


def _serve_point(zarr_path):
    """Drive a real /api/tide point query against `zarr_path` through the
    app lifespan (adapter open + predict)."""
    from fastapi.testclient import TestClient

    os.chdir(REPO_ROOT)
    os.environ["TIDE_ZARR_PATH"] = str(zarr_path)
    os.environ["TIDE_DASK_DISABLE"] = "1"   # fast startup; no scheduler in tests
    try:
        import importlib

        import tide_app
        importlib.reload(tide_app)
        with TestClient(tide_app.app) as client:
            resp = client.get(
                "/api/tide",
                params={"lon0": 125.0, "lat0": 15.0,
                        "start": "2023-07-25", "end": "2023-07-26"},
            )
        return resp
    finally:
        os.environ.pop("TIDE_ZARR_PATH", None)


@pytest.mark.skipif(not TPXO10_STORE.exists(), reason="tpxo10.zarr not present")
def test_api_serves_tpxo10_default():
    resp = _serve_point(TPXO10_STORE)
    assert resp.status_code == 200 and resp.json()


@pytest.mark.skipif(not TPXO9_STORE.exists(), reason="tpxo9.zarr not present")
def test_api_serves_tpxo9_rollback():
    """Rollback: the same migrated runtime serves the legacy store when
    TIDE_ZARR_PATH points back at it (D9 schema adapter)."""
    resp = _serve_point(TPXO9_STORE)
    assert resp.status_code == 200 and resp.json()
