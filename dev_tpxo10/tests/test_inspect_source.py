"""Unit tests for inspect_source helpers (no big data required)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from inspect_source import (  # noqa: E402
    DELTA,
    check_node_offsets,
    constituent_from_filename,
    expected_files,
    parse_con,
)


def _toy_grid(nx=10800, ny=5401, delta=DELTA):
    lon_z = delta / 2.0 + delta * np.arange(nx)
    lat_z = -90.0 + delta * np.arange(ny)
    return (
        lon_z,
        lat_z,
        lon_z - delta / 2.0,  # lon_u: western edge
        lat_z.copy(),         # lat_u
        lon_z.copy(),         # lon_v
        lat_z - delta / 2.0,  # lat_v: southern edge
    )


def test_expected_files_inventory():
    files = expected_files()
    assert len(files) == 31
    assert files[0] == "grid_tpxo10atlas_v2.nc"
    assert sum(f.startswith("h_") for f in files) == 15
    assert sum(f.startswith("u_") for f in files) == 15


def test_node_offsets_pass_on_conventional_grid():
    assert check_node_offsets(*_toy_grid()) == []


def test_node_offsets_detect_eastern_edge_u():
    lon_z, lat_z, _, lat_u, lon_v, lat_v = _toy_grid()
    lon_u_wrong = lon_z + DELTA / 2.0  # eastern edge: convention flipped
    errs = check_node_offsets(lon_z, lat_z, lon_u_wrong, lat_u, lon_v, lat_v)
    assert any("western edge" in e for e in errs)


def test_node_offsets_detect_nonperiodic_span():
    grid = _toy_grid(nx=10000)  # 10000 * (1/30) != 360
    errs = check_node_offsets(*grid)
    assert any("periodic" in e for e in errs)


def test_node_offsets_detect_nonuniform_spacing():
    lon_z, lat_z, lon_u, lat_u, lon_v, lat_v = _toy_grid()
    lat_z2 = lat_z.copy()
    lat_z2[2700] += 0.01  # the historical tpxo9 lat-corruption shape
    errs = check_node_offsets(lon_z, lat_z2, lon_u, lat_u, lon_v, lat_v)
    assert any("lat_z spacing" in e for e in errs)


def test_node_offsets_detect_tiny_offset_at_high_longitude():
    """Round 8 finding 1 regression: with NumPy's default rtol, a 1e-4 deg
    u-node misalignment near lon=360 would pass (tolerance there would be
    ~3.6e-3 deg). With rtol=0.0 it must fail."""
    lon_z, lat_z, lon_u, lat_u, lon_v, lat_v = _toy_grid()
    lon_u2 = lon_u.copy()
    lon_u2[-1] += 1e-4  # last column, lon ~ 359.97 deg
    errs = check_node_offsets(lon_z, lat_z, lon_u2, lat_u, lon_v, lat_v)
    assert any("western edge" in e for e in errs)


def test_parse_con_and_filename_constituent():
    con = np.array([b"m", b"2", b" ", b" "], dtype="S1")
    assert parse_con(con) == "m2"
    con4 = np.array([b"2", b"n", b"2", b" "], dtype="S1")
    assert parse_con(con4) == "2n2"
    assert constituent_from_filename("h_m2_tpxo10_atlas_30_v2.nc") == "m2"
    assert constituent_from_filename("u_2n2_tpxo10_atlas_30_v2.nc") == "2n2"
