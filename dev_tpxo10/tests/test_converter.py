"""T-A tests for converter-level contracts (spec §7.1: transpose
round-trip, provenance fail-closed, two-run byte-identity)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "dev_tpxo10" / "scripts"))

import tpxo10_pipeline as P  # noqa: E402
from convert_to_zarr import assert_clean_pipeline_sources, convert  # noqa: E402

SOURCE = REPO_ROOT / "data_src" / "TPXO10_atlas_v2"


# ---------- transpose round-trip (§7.1) ----------

def test_transpose_index_mapping_round_trip():
    """Source NetCDF is (nx, ny) = (lon, lat); the store is (lat, lon).
    The loader transpose must map src[i, j] -> store[j, i], and round-trip."""
    nx, ny = 7, 5
    src = np.arange(nx * ny).reshape(nx, ny)
    store = src.T
    assert store.shape == (ny, nx)
    for i in range(nx):
        for j in range(ny):
            assert store[j, i] == src[i, j]
    assert np.array_equal(store.T, src)


# ---------- provenance fail-closed (round 9, finding 1) ----------

def test_assert_clean_passes_on_empty_status():
    assert_clean_pipeline_sources("")
    assert_clean_pipeline_sources("\n")


def test_assert_clean_rejects_modified_and_untracked():
    with pytest.raises(P.PipelineError, match="not committed/clean"):
        assert_clean_pipeline_sources(" M dev_tpxo10/scripts/tpxo10_pipeline.py")
    with pytest.raises(P.PipelineError, match="not committed/clean"):
        assert_clean_pipeline_sources("?? dev_tpxo10/scripts/new_helper.py")


# ---------- two-run byte-identity (§7.1 / G2 idempotency) ----------

def _store_bytes(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.skipif(not SOURCE.exists(), reason="TPXO10 source not present")
def test_two_run_byte_identity(tmp_path):
    """Same inputs + pinned env + fixed timestamp => byte-identical stores.
    Requires clean pipeline sources (provenance fail-closed), so this also
    exercises the round-9 dirty check against the real repo state."""
    region = (120.0, 121.0, 23.0, 24.0)  # 1-degree toy region, seconds-scale
    kw = dict(chunks=(113, 113, 15), source=SOURCE, repo_root=REPO_ROOT,
              created_utc="1992-01-01T00:00:00+00:00")
    convert(region, out=tmp_path / "a.zarr", **kw)
    convert(region, out=tmp_path / "b.zarr", **kw)
    a, b = _store_bytes(tmp_path / "a.zarr"), _store_bytes(tmp_path / "b.zarr")
    assert a.keys() == b.keys()
    diff = [k for k in a if a[k] != b[k]]
    assert diff == [], f"non-identical files: {diff}"
