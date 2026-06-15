"""v0.3.0 Stage 3 unified bbox/map query planner (spec §7.5.3 + review
round 19 F7).

ONE selection/enforcement path for all bbox/map /api/tide queries (scalar
point and vectorized multipoint go through the adapter directly and are
not cell-capped), fixing the order so the cap runs BEFORE any
materialization:

    resolve INDEX arrays from coordinates -> apply sample -> count output
      cells -> enforce MAX_BBOX_CELLS (HTTP 400)
      -> isel (still lazy) -> materialize hc -> predict -> serialize

The cap (`MAX_BBOX_CELLS`, owner+reviewer signed 500_000, env-overridable)
is counted on the POST-sample OUTPUT grid (`ceil(nlat/sample) *
ceil(nlon/sample)`), so a default `sample>=5` request is unaffected and
only genuinely oversized outputs are rejected. Counting uses ONLY the
coordinate index arrays — never the data arrays, and never an eager
concat — so a rejected request (incl. a dateline-wrap request) reads
ZERO data chunks (round 22 F1).
"""
from __future__ import annotations

import math
import operator
import os
from typing import Iterable, Optional

DEFAULT_MAX_BBOX_CELLS = 500_000


def _as_sample(sample) -> int:
    """Fail-closed sample coercion (round 22 F2): integer >= 1; reject
    bool, float and non-positive values."""
    if isinstance(sample, bool):
        raise ValueError(f"sample must be an integer, not bool ({sample!r})")
    try:
        s = operator.index(sample)
    except TypeError as e:
        raise ValueError(f"sample must be an integer, got {sample!r}") from e
    if s < 1:
        raise ValueError(f"sample must be >= 1, got {s}")
    return s


def get_max_bbox_cells(default: int = DEFAULT_MAX_BBOX_CELLS) -> int:
    """Resolve the cap: `TIDE_MAX_BBOX_CELLS` env override, else the
    v0.3.0 signed default (500_000)."""
    env = os.environ.get("TIDE_MAX_BBOX_CELLS")
    if env is None:
        return default
    try:
        val = int(env)
    except ValueError as e:
        raise ValueError(f"TIDE_MAX_BBOX_CELLS={env!r} is not an integer") from e
    if val <= 0:
        raise ValueError(f"TIDE_MAX_BBOX_CELLS must be positive, got {val}")
    return val


def output_cell_count(nlat: int, nlon: int, sample) -> int:
    """Output cell count after `sample`-stride decimation (isel
    slice(None, None, sample))."""
    s = _as_sample(sample)
    return math.ceil(nlat / s) * math.ceil(nlon / s)


class BboxCapError(ValueError):
    """Requested output exceeds MAX_BBOX_CELLS (maps to HTTP 400). Carries
    the actual cell count and the limit for a descriptive message."""

    def __init__(self, requested: int, cap: int, nlat: int, nlon: int,
                 sample: int):
        self.requested = requested
        self.cap = cap
        self.nlat, self.nlon, self.sample = nlat, nlon, sample
        super().__init__(
            f"requested map of {requested} output cells "
            f"({nlat}x{nlon} at sample={sample}) exceeds MAX_BBOX_CELLS={cap}; "
            "increase `sample` or shrink the bbox")


class EmptyBboxError(ValueError):
    """A bbox selecting zero cells (reversed/out-of-range bounds) — maps to
    HTTP 400 (round 22 F3)."""


def plan_bbox(adapter, lon0: float, lon1: float, lat0: float, lat1: float,
              sample, constituents: Optional[Iterable] = None,
              halo: float = 0.0, max_cells: Optional[int] = None):
    """Plan a bbox map query up to (but not including) materialization.
    Returns (lazy_subset, output_cells).

    Resolves the post-sample INDEX arrays from coordinates (no data read,
    no eager concat); rejects an empty selection (EmptyBboxError); counts
    output cells and enforces the cap (BboxCapError) — all BEFORE the
    isel. On rejection NO data chunk is read, including dateline-wrap
    requests (round 22 F1)."""
    if max_cells is None:
        max_cells = get_max_bbox_cells()
    s = _as_sample(sample)
    lat_idx, lon_idx = adapter.bbox_indices(lon0, lon1, lat0, lat1,
                                            sample=s, halo=halo)
    nlat, nlon = len(lat_idx), len(lon_idx)
    if nlat == 0 or nlon == 0:
        raise EmptyBboxError(
            f"bbox selects zero cells (lat {nlat}, lon {nlon}); check that "
            "lat0<=lat1 and the bbox overlaps the data extent")
    cells = nlat * nlon
    if cells > max_cells:
        raise BboxCapError(cells, max_cells, nlat, nlon, s)
    sub = adapter.isel_grid(lat_idx, lon_idx, constituents=constituents)
    return sub, cells
