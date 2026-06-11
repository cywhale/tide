"""TPXO10 -> Zarr conversion pipeline core (spec v0.3.0 Stage 1+2).

Pure, unit-testable functions implementing the frozen conversion contract
(G1 kickoff sign-off 2026-06-11):

* two-layer store (§3.0): bit-exact source layer (flag 0) + deterministic
  derived fill layer (flag 1); land/invalid (flag 2, value 0)
* §3.2 validity rule (provisional, G1-calibrated): node valid iff its
  bathymetry > 0 AND not all constituents are (0+0j)
* bounded inpaint: pyTMD.interpolate.inpaint(N=0) (nearest-neighbor mode,
  Garcia 2010 implementation; D3), applied per constituent per component,
  written back ONLY to cells within DMAX_FILL_CELLS of valid data
* quantization (§3.0): rint (ties-to-even) -> finite/int32-range check on
  the ROUNDED value -> cast; abort on violation, no silent clip/wrap
* D12 centering: edge velocities first (1e-4 * U / h_edge, NO depth clamp,
  h <= 0 invalid), then two-edge average onto z-nodes; flag precedence:
  0 = both edges native, 1 = one-sided or any inpainted edge, 2 = none
* D12 regional rule: compute on haloed windows read from the global
  source, write interior only, never wrap locally at a regional edge
"""
from __future__ import annotations

import numpy as np
import numpy.ma as ma
from scipy.ndimage import distance_transform_edt

import pyTMD.interpolate

DELTA = 1.0 / 30.0
NX, NY = 10800, 5401
CONSTITUENTS = [
    "2n2", "k1", "k2", "m2", "m4", "mf", "mm", "mn4",
    "ms4", "n2", "o1", "p1", "q1", "s1", "s2",
]
# Frozen Stage 1 parameters (decision memo -> spec at G1 freeze)
DMAX_FILL_CELLS = 8        # inpaint band: fill only within 8 cells of valid data
HALO_CELLS = 32            # regional halo; >= DMAX + centering(+1) with margin
QUANTIZATION_VERSION = "rint-ties-even-v1"
INPAINT_PARAMS = {"N": 0}  # nearest-neighbor mode (deterministic)
SCHEMA_VERSION = "tpxo10-cgrid-v1"
TRANSPORT_SCALE = 1e-4     # cm^2/s -> m^2/s (per-edge, divided by edge depth)

FLAG_SOURCE, FLAG_FILLED, FLAG_INVALID = 0, 1, 2
INT32_MIN, INT32_MAX = np.iinfo(np.int32).min, np.iinfo(np.int32).max


class PipelineError(RuntimeError):
    """Deterministic-contract violation: always abort, never salvage."""


def region_to_index_window(
    lon0: float, lon1: float, lat0: float, lat1: float,
    lon_z: np.ndarray, lat_z: np.ndarray,
) -> tuple[slice, slice]:
    """Output-region index window on the z-axis index space (shared by
    u/v nodes: the C-grid relation is index-aligned, D12)."""
    i0 = int(np.searchsorted(lon_z, lon0, side="left"))
    i1 = int(np.searchsorted(lon_z, lon1, side="right"))
    j0 = int(np.searchsorted(lat_z, lat0, side="left"))
    j1 = int(np.searchsorted(lat_z, lat1, side="right"))
    if i0 >= i1 or j0 >= j1:
        raise PipelineError(f"empty region window: lon[{i0}:{i1}] lat[{j0}:{j1}]")
    return slice(j0, j1), slice(i0, i1)


def haloed_window(
    j_int: slice, i_int: slice, halo: int = HALO_CELLS,
    ny: int = NY, nx: int = NX,
) -> tuple[slice, slice]:
    """Halo extension. Stage 1 regional rule: the halo must stay inside
    the global grid (no wrap at an artificial regional edge — D12). Global
    runs (Stage 2) use the full grid and periodic wrap instead."""
    j0, j1 = j_int.start - halo, j_int.stop + halo
    i0, i1 = i_int.start - halo, i_int.stop + halo
    if j0 < 0 or j1 > ny or i0 < 0 or i1 > nx:
        raise PipelineError(
            f"halo window lat[{j0}:{j1}] lon[{i0}:{i1}] exceeds the global "
            f"grid; regional wrap is forbidden (D12 regional rule)"
        )
    return slice(j0, j1), slice(i0, i1)


def compute_validity(h: np.ndarray, re: np.ndarray, im: np.ndarray) -> np.ndarray:
    """§3.2 rule: valid iff h > 0 AND not all constituents are (0+0j).
    h: (nlat, nlon); re/im: (nlat, nlon, ncons) int32."""
    all_zero = np.all((re == 0) & (im == 0), axis=-1)
    return (h > 0) & ~all_zero


def classify_flags(h: np.ndarray, valid: np.ndarray,
                   dmax: int = DMAX_FILL_CELLS) -> np.ndarray:
    """flag 0 = source-valid; flag 1 = invalid-but-fillable (h > 0, within
    dmax cells of valid data); flag 2 = land / unfillable."""
    flag = np.full(h.shape, FLAG_INVALID, dtype=np.uint8)
    flag[valid] = FLAG_SOURCE
    dist = distance_transform_edt(~valid)
    flag[(h > 0) & ~valid & (dist <= dmax)] = FLAG_FILLED
    return flag


def quantize_int32(values: np.ndarray, context: str = "") -> np.ndarray:
    """§3.0 quantization, order pinned by review round 7: (1) rint
    (ties-to-even), (2) check the ROUNDED value finite + int32 range,
    (3) cast. Abort on violation."""
    rounded = np.rint(np.asarray(values, dtype=np.float64))
    bad = ~np.isfinite(rounded)
    if np.any(bad):
        raise PipelineError(
            f"quantize[{context}]: {int(bad.sum())} non-finite rounded values"
        )
    out_of_range = (rounded < INT32_MIN) | (rounded > INT32_MAX)
    if np.any(out_of_range):
        raise PipelineError(
            f"quantize[{context}]: {int(out_of_range.sum())} values outside "
            f"int32 range (max |rounded| = {np.abs(rounded).max():.6g})"
        )
    return rounded.astype(np.int32)


def inpaint_fill_inplace(
    lon_axis: np.ndarray, lat_axis: np.ndarray,
    re: np.ndarray, im: np.ndarray,
    valid: np.ndarray, flag: np.ndarray,
    context: str = "",
) -> None:
    """Fill flag==1 cells of re/im (int32, (nlat, nlon, ncons)) in place,
    per constituent per component (independent float64 inpaint), using
    pyTMD.interpolate.inpaint with frozen INPAINT_PARAMS, then quantize.
    Cells with flag != 1 are never modified (asserted by T-B)."""
    fill = flag == FLAG_FILLED
    if not np.any(fill):
        return
    for k in range(re.shape[-1]):
        for name, comp in (("Re", re), ("Im", im)):
            arr = comp[..., k].astype(np.float64)
            arr[~valid] = np.nan
            out = np.asarray(
                pyTMD.interpolate.inpaint(
                    lon_axis, lat_axis, ma.masked_invalid(arr), **INPAINT_PARAMS
                )
            )
            comp[..., k][fill] = quantize_int32(
                out[fill], context=f"{context}/{CONSTITUENTS[k]}/{name}"
            )


def edge_velocity(
    tr_re: np.ndarray, tr_im: np.ndarray, h_edge: np.ndarray, flag: np.ndarray
) -> np.ndarray:
    """Per-edge velocity (m/s, complex128): 1e-4 * (Re + i*Im) / h_edge for
    usable edges (flag <= 1; h > 0 by construction). NO depth clamp (G1
    sign-off: h <= 0 is invalid; positive depth divides as-is). Unusable
    edges become NaN."""
    usable = flag <= FLAG_FILLED
    h = np.where(usable, h_edge, np.nan)[..., np.newaxis]
    vel = TRANSPORT_SCALE * (tr_re + 1j * tr_im) / h
    vel[~usable] = np.nan
    return vel


def _center_pair(e0, e1, f0, f1):
    """Common D12 two-edge centering given the two edge stacks and flags.
    Returns (centered complex, flag uint8) with the round-4 precedence."""
    u0, u1 = f0 <= FLAG_FILLED, f1 <= FLAG_FILLED
    both, only0, only1 = u0 & u1, u0 & ~u1, ~u0 & u1
    val = np.zeros(e0.shape, dtype=np.complex128)
    val[both] = 0.5 * (e0[both] + e1[both])
    val[only0] = e0[only0]
    val[only1] = e1[only1]
    flag = np.full(f0.shape, FLAG_INVALID, dtype=np.uint8)
    flag[both | only0 | only1] = FLAG_FILLED
    flag[both & (f0 == FLAG_SOURCE) & (f1 == FLAG_SOURCE)] = FLAG_SOURCE
    return val, flag


def center_u(eu: np.ndarray, u_flag: np.ndarray, wrap: bool):
    """uz[j,i] from western edge eu[j,i] and eastern edge eu[j,i+1] (D12;
    Stage 0-asserted convention lon_u = lon_z - delta/2). wrap=True only
    for the full global grid (periodic longitude); wrap=False returns
    nlon-1 columns (caller supplies a haloed array and crops)."""
    if wrap:
        east = np.roll(eu, -1, axis=1)
        f_east = np.roll(u_flag, -1, axis=1)
        return _center_pair(
            eu, east,
            np.repeat(u_flag[..., np.newaxis], eu.shape[-1], axis=-1),
            np.repeat(f_east[..., np.newaxis], eu.shape[-1], axis=-1),
        )
    return _center_pair(
        eu[:, :-1, :], eu[:, 1:, :],
        np.repeat(u_flag[:, :-1, np.newaxis], eu.shape[-1], axis=-1),
        np.repeat(u_flag[:, 1:, np.newaxis], eu.shape[-1], axis=-1),
    )


def center_v(ev: np.ndarray, v_flag: np.ndarray, last_row_one_sided: bool = False):
    """vz[j,i] from southern edge ev[j,i] and northern edge ev[j+1,i]
    (D12; lat_v = lat_z - delta/2). Latitude is non-periodic: the last
    available row centers one-sided ONLY at the true global boundary
    (last_row_one_sided=True, Stage 2); regional windows must instead be
    haloed so the northern neighbor exists (returns nlat-1 rows)."""
    val, flag = _center_pair(
        ev[:-1, :, :], ev[1:, :, :],
        np.repeat(v_flag[:-1, :, np.newaxis], ev.shape[-1], axis=-1),
        np.repeat(v_flag[1:, :, np.newaxis], ev.shape[-1], axis=-1),
    )
    if last_row_one_sided:
        last_val = np.zeros((1,) + ev.shape[1:], dtype=np.complex128)
        usable = (v_flag[-1:, :] <= FLAG_FILLED)[..., np.newaxis]
        usable = np.repeat(usable, ev.shape[-1], axis=-1)
        last_val[usable] = ev[-1:, :, :][usable]
        last_flag = np.where(usable, FLAG_FILLED, FLAG_INVALID).astype(np.uint8)
        val = np.concatenate([val, last_val], axis=0)
        flag = np.concatenate([flag, last_flag], axis=0)
    return val, flag


def collapse_constituent_flags(flag_3d: np.ndarray, context: str = "") -> np.ndarray:
    """uz/vz flags are per-node (2D) in the §3.1 schema. Edge usability is
    constituent-independent (validity is an all-constituent property), so
    all constituent slices must agree — asserted here, then collapsed."""
    first = flag_3d[..., 0]
    if not np.all(flag_3d == first[..., np.newaxis]):
        raise PipelineError(f"{context}: centering flags differ across constituents")
    return first.astype(np.uint8)
