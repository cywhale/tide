"""Unit tests for tpxo10_pipeline (spec §7.1 conversion side; toy grids only)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import tpxo10_pipeline as P  # noqa: E402


# ---------- quantization (§3.0, rounds 6-7) ----------

def test_quantize_ties_to_even():
    out = P.quantize_int32(np.array([0.5, 1.5, 2.5, -0.5, -1.5]))
    assert out.tolist() == [0, 2, 2, 0, -2]


def test_quantize_negative_and_boundary_roundable():
    # near-boundary values that ROUND to a representable int32 must pass
    out = P.quantize_int32(np.array([2147483646.4, -2147483647.6]))
    assert out.tolist() == [2147483646, -2147483648]


def test_quantize_overflow_aborts():
    with pytest.raises(P.PipelineError, match="outside int32 range"):
        P.quantize_int32(np.array([2.2e9]))


def test_quantize_nonfinite_aborts():
    with pytest.raises(P.PipelineError, match="non-finite"):
        P.quantize_int32(np.array([np.nan]))
    with pytest.raises(P.PipelineError, match="non-finite"):
        P.quantize_int32(np.array([np.inf]))


# ---------- validity + flags (§3.2 / D3) ----------

def _toy_field(nlat=20, nlon=30, ncons=3, seed=7):
    rng = np.random.default_rng(seed)
    re = rng.integers(-1000, 1000, size=(nlat, nlon, ncons)).astype(np.int32)
    im = rng.integers(-1000, 1000, size=(nlat, nlon, ncons)).astype(np.int32)
    h = np.full((nlat, nlon), 50.0, dtype=np.float32)
    return h, re, im


def test_validity_rule():
    h, re, im = _toy_field()
    h[0, 0] = 0.0                      # land: h <= 0
    re[1, 1, :], im[1, 1, :] = 0, 0    # ocean-by-bathymetry, all-zero hc
    re[2, 2, 0], im[2, 2, 0] = 0, 0    # single zero constituent: still valid
    valid = P.compute_validity(h, re, im)
    assert not valid[0, 0]
    assert not valid[1, 1]
    assert valid[2, 2]


def test_flags_band_and_land():
    h, re, im = _toy_field()
    h[:, 0] = 0.0                       # land column
    re[5:8, 10:13, :], im[5:8, 10:13, :] = 0, 0   # small hole: fillable
    valid = P.compute_validity(h, re, im)
    flag = P.classify_flags(h, valid, dmax=8)
    assert (flag[:, 0] == P.FLAG_INVALID).all()
    assert (flag[5:8, 10:13] == P.FLAG_FILLED).all()
    assert flag[0, 5] == P.FLAG_SOURCE


def test_flags_far_hole_stays_invalid():
    nlat, nlon = 40, 60
    h = np.full((nlat, nlon), 50.0, dtype=np.float32)
    re = np.ones((nlat, nlon, 2), dtype=np.int32)
    im = np.ones_like(re)
    re[10:30, 10:50, :], im[10:30, 10:50, :] = 0, 0   # huge hole
    valid = P.compute_validity(h, re, im)
    flag = P.classify_flags(h, valid, dmax=3)
    assert flag[20, 30] == P.FLAG_INVALID   # deep inside: > dmax from valid
    assert flag[10, 10] == P.FLAG_FILLED    # rim: within dmax


# ---------- inpaint (D3: deterministic, fill-only, quantized) ----------

def test_inpaint_fills_only_flagged_cells_and_is_deterministic():
    h, re, im = _toy_field()
    re[5:7, 10:12, :], im[5:7, 10:12, :] = 0, 0
    valid = P.compute_validity(h, re, im)
    flag = P.classify_flags(h, valid)
    lon = np.arange(re.shape[1], dtype=float)
    lat = np.arange(re.shape[0], dtype=float)

    re1, im1 = re.copy(), im.copy()
    P.inpaint_fill_inplace(lon, lat, re1, im1, valid, flag)
    re2, im2 = re.copy(), im.copy()
    P.inpaint_fill_inplace(lon, lat, re2, im2, valid, flag)

    assert np.array_equal(re1, re2) and np.array_equal(im1, im2)  # byte-identity
    untouched = flag != P.FLAG_FILLED
    assert np.array_equal(re1[untouched], re[untouched])          # source layer intact
    assert (re1[flag == P.FLAG_FILLED] != 0).any()                # actually filled
    assert re1.dtype == np.int32


# ---------- edge velocity (G1: no clamp, h<=0 invalid) ----------

def test_edge_velocity_no_clamp_and_invalid_nan():
    tr_re = np.full((2, 2, 1), 10000, dtype=np.int32)   # 1 m^2/s after scale
    tr_im = np.zeros_like(tr_re)
    h = np.array([[2.0, 0.5], [50.0, 1.0]], dtype=np.float32)
    flag = np.zeros((2, 2), dtype=np.uint8)
    flag[1, 1] = P.FLAG_INVALID
    vel = P.edge_velocity(tr_re, tr_im, h, flag)
    assert np.isclose(vel[0, 0, 0].real, 0.5)    # 1 m^2/s / 2 m
    assert np.isclose(vel[0, 1, 0].real, 2.0)    # shallow: divides as-is, no clamp
    assert np.isnan(vel[1, 1, 0])                # invalid edge -> NaN


# ---------- D12 centering ----------

def _edges(vals, flags):
    e = np.asarray(vals, dtype=np.complex128)[..., np.newaxis]
    f = np.asarray(flags, dtype=np.uint8)
    e[f[..., np.newaxis] > P.FLAG_FILLED] = np.nan
    return e, f


def test_center_u_two_edge_one_sided_invalid_and_inpaint_precedence():
    # one row, 5 u-edges -> 4 z-cells
    eu, fu = _edges([[1.0, 3.0, 5.0, 7.0, 9.0]],
                    [[0, 0, 1, 2, 0]])
    val, flag = P.center_u(eu, fu, wrap=False)
    assert np.isclose(val[0, 0, 0].real, 2.0) and flag[0, 0, 0] == 0   # native+native
    assert np.isclose(val[0, 1, 0].real, 4.0) and flag[0, 1, 0] == 1   # native+inpainted (round 4)
    assert np.isclose(val[0, 2, 0].real, 5.0) and flag[0, 2, 0] == 1   # one-sided (west only)
    assert np.isclose(val[0, 3, 0].real, 9.0) and flag[0, 3, 0] == 1   # one-sided (east only)


def test_center_u_no_usable_edges():
    eu, fu = _edges([[1.0, 1.0]], [[2, 2]])
    val, flag = P.center_u(eu, fu, wrap=False)
    assert flag[0, 0, 0] == P.FLAG_INVALID and val[0, 0, 0] == 0


def test_center_u_periodic_wrap_global():
    # global mode: last z-cell uses edges [i=last, i=0] via wrap
    eu, fu = _edges([[2.0, 4.0, 6.0]], [[0, 0, 0]])
    val, flag = P.center_u(eu, fu, wrap=True)
    assert val.shape[1] == 3
    assert np.isclose(val[0, 2, 0].real, 0.5 * (6.0 + 2.0))   # wraps to column 0
    assert flag[0, 2, 0] == 0


def test_center_v_nonperiodic_one_sided_last_row():
    ev, fv = _edges([[1.0], [3.0], [5.0]], [[0], [0], [0]])
    val, flag = P.center_v(ev, fv, last_row_one_sided=True)
    assert val.shape[0] == 3
    assert np.isclose(val[0, 0, 0].real, 2.0) and flag[0, 0, 0] == 0
    assert np.isclose(val[2, 0, 0].real, 5.0) and flag[2, 0, 0] == 1   # boundary one-sided


def test_regional_halo_interior_matches_global_toy():
    """Round 4: halo-computed interior must equal the global computation."""
    rng = np.random.default_rng(3)
    nlat, nlon, ncons = 12, 16, 2
    eu = rng.normal(size=(nlat, nlon, ncons)) + 1j * rng.normal(size=(nlat, nlon, ncons))
    fu = rng.integers(0, 3, size=(nlat, nlon)).astype(np.uint8)
    eu = eu.copy()
    eu[np.repeat((fu > 1)[..., None], ncons, axis=-1)] = np.nan

    g_val, g_flag = P.center_u(eu, fu, wrap=True)
    # regional window [rows 2:10, cols 3:11] with halo 2 (no wrap involved)
    h_val, h_flag = P.center_u(eu[0:12, 1:14, :], fu[0:12, 1:14], wrap=False)
    # interior of the haloed result: cols 3:11 map to haloed-output cols 2:10
    assert np.array_equal(np.nan_to_num(h_val[2:10, 2:10, :]),
                          np.nan_to_num(g_val[2:10, 3:11, :]))
    assert np.array_equal(h_flag[2:10, 2:10, :], g_flag[2:10, 3:11, :])


def test_collapse_constituent_flags_asserts_agreement():
    f = np.zeros((2, 2, 3), dtype=np.uint8)
    assert P.collapse_constituent_flags(f).shape == (2, 2)
    f[0, 0, 1] = 1
    with pytest.raises(P.PipelineError, match="differ across constituents"):
        P.collapse_constituent_flags(f)


def _full_chain(h, re, im, jh0, jh1, top, dmax, lon_ax, lat_ax):
    """The convert_full per-tile computation chain on a haloed row window
    (mirrors convert_to_zarr.convert_full's tile body)."""
    hw = h[jh0:jh1]
    rew, imw = re[jh0:jh1].copy(), im[jh0:jh1].copy()
    valid = P.compute_validity(hw, rew, imw)
    flag = P.classify_flags(hw, valid, dmax=dmax)
    P.inpaint_fill_inplace(lon_ax, lat_ax[jh0:jh1], rew, imw, valid, flag)
    inv = flag == P.FLAG_INVALID
    rew[inv], imw[inv] = 0, 0
    evel = P.edge_velocity(rew, imw, hw, flag)
    uz, uzf = P.center_u(evel, flag, wrap=True)
    vz, vzf = P.center_v(evel, flag, last_row_one_sided=top)
    return rew, imw, flag, uz, uzf, vz, vzf


def _synthetic_globe(ny=24, nx=36, ncons=2, seed=11):
    """Periodic-lon toy globe with: an all-land Antarctic band, a land
    strip crossing the dateline (one-sided wrap centering), coastal
    notch holes with UNIQUE nearest donors (no inpaint tie-break
    ambiguity) — one ON a tile seam row, one on the last (polar) row."""
    rng = np.random.default_rng(seed)
    h = np.full((ny, nx), 50.0, dtype=np.float32)
    re = rng.integers(-900, 900, size=(ny, nx, ncons)).astype(np.int32)
    im = rng.integers(-900, 900, size=(ny, nx, ncons)).astype(np.int32)
    re[re == 0] = 7
    im[im == 0] = -7
    h[0:2, :] = 0.0                       # antarctic land band
    h[:, 34] = 0.0                        # dateline-adjacent land column
    h[10:14, 8:12] = 0.0                  # inland lake/land block
    # notch holes: land on three sides, unique ocean donor on the fourth
    for (j, i) in [(7, 20), (ny - 1, 5)]:  # seam row (tile=7), polar row
        h[j, i] = 30.0
        re[j, i, :], im[j, i, :] = 0, 0    # all-zero hc -> fillable
        if j < ny - 1:
            h[j + 1, i] = 0.0
        h[j, i - 1] = 0.0
        if j > 0:
            h[j - 1, i] = 0.0              # unique donor at (j, i+1)
    return h, re, im


def test_tiled_pipeline_bitwise_matches_monolithic_globe():
    """Round 15, finding 2: tile seams, dateline periodic centering, the
    one-sided polar row and a seam-crossing fill band must all be
    bit-identical between the tiled (convert_full) decomposition and a
    monolithic computation."""
    ny, nx, dmax, halo, tile = 24, 36, 2, 4, 7
    h, re, im = _synthetic_globe(ny=ny, nx=nx)
    lon_ax = np.arange(nx, dtype=float)
    lat_ax = np.arange(ny, dtype=float)

    mono = _full_chain(h, re, im, 0, ny, True, dmax, lon_ax, lat_ax)
    m_re, m_im, m_flag, m_uz, m_uzf, m_vz, m_vzf = mono
    # sanity: the synthetic scenario actually exercises every branch
    assert m_flag[7, 20] == P.FLAG_FILLED          # seam-row fill
    assert m_flag[ny - 1, 5] == P.FLAG_FILLED      # polar-row fill
    # dateline: land at lon 34 makes col 34 one-sided, while col 35 (last)
    # wraps to edge 0 with BOTH edges valid -> the periodic path is hit
    assert (m_uzf[5, 34, 0] == P.FLAG_FILLED)
    assert (m_uzf[5, 35, 0] == P.FLAG_SOURCE)
    assert (m_vzf[ny - 1, :, 0] <= P.FLAG_FILLED).any()  # one-sided polar row

    for (j0, j1, jh0, jh1, top) in P.lat_tiles(ny=ny, tile=tile, halo=halo):
        sl = slice(j0 - jh0, j0 - jh0 + (j1 - j0))
        t_re, t_im, t_flag, t_uz, t_uzf, t_vz, t_vzf = _full_chain(
            h, re, im, jh0, jh1, top, dmax, lon_ax, lat_ax)
        gsl = slice(j0, j1)
        assert np.array_equal(t_re[sl], m_re[gsl])
        assert np.array_equal(t_im[sl], m_im[gsl])
        assert np.array_equal(t_flag[sl], m_flag[gsl])
        assert np.array_equal(t_uz[sl], m_uz[gsl])
        assert np.array_equal(t_uzf[sl], m_uzf[gsl])
        assert np.array_equal(t_vz[sl], m_vz[gsl])
        assert np.array_equal(t_vzf[sl], m_vzf[gsl])


def test_lat_tiles_cover_globe_without_overlap():
    tiles = P.lat_tiles(ny=5401, tile=226, halo=32)
    assert tiles[0][0] == 0 and tiles[-1][1] == 5401
    for (a, b) in zip(tiles, tiles[1:]):
        assert a[1] == b[0]                      # contiguous, no overlap
    assert sum(t[1] - t[0] for t in tiles) == 5401
    assert [t[4] for t in tiles] == [False] * (len(tiles) - 1) + [True]
    for j0, j1, jh0, jh1, _ in tiles:
        assert jh0 == max(0, j0 - 32) and jh1 == min(5401, j1 + 32)
        assert jh0 >= 0 and jh1 <= 5401          # clamped at global edges


# ---------- window helpers ----------

def test_haloed_window_rejects_global_overflow():
    with pytest.raises(P.PipelineError, match="regional wrap is forbidden"):
        P.haloed_window(slice(10, 100), slice(5, 50), halo=32)
    j, i = P.haloed_window(slice(100, 200), slice(100, 200), halo=32)
    assert (j.start, j.stop, i.start, i.stop) == (68, 232, 68, 232)
