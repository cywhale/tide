"""Stage 0 G0 smoke tests (spec v0.3.0 §4 Gate G0).

Verifies the existing app imports and still serves from data/tpxo9.zarr,
unchanged, inside the uv-managed production environment.
"""
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TPXO9_STORE = REPO_ROOT / "data" / "tpxo9.zarr"


def test_app_importable():
    import tide_app

    assert tide_app.app is not None


def test_pinned_runtime_versions():
    import pyTMD
    import zarr

    assert pyTMD.version.full_version == "v2.2.8"
    assert zarr.__version__ == "2.18.7"


@pytest.mark.skipif(not TPXO9_STORE.exists(), reason="tpxo9.zarr not present")
def test_api_serves_from_tpxo9_zarr():
    from fastapi.testclient import TestClient

    os.chdir(REPO_ROOT)  # lifespan opens the store via a relative path
    import tide_app

    with TestClient(tide_app.app) as client:
        resp = client.get(
            "/api/tide",
            params={
                "lon0": 125.0,
                "lat0": 15.0,
                "start": "2023-07-25",
                "end": "2023-07-26",
            },
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload  # non-empty response from the legacy store
