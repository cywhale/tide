"""v0.3.0 Stage 3 unified query planner (spec §7.5.3 + review round 19 F7).

ONE selection/enforcement path for every /api/tide query shape, fixing
the order so the protective cell cap runs BEFORE any materialization:

    resolve indexes -> apply sample -> count output cells
      -> enforce MAX_BBOX_CELLS (HTTP 400)
      -> select arrays (still lazy) -> materialize hc -> predict -> serialize

The cap (`MAX_BBOX_CELLS`, owner+reviewer signed 500_000, env-overridable)
is counted on the POST-sample OUTPUT grid (`ceil(nlat/sample) *
ceil(nlon/sample)`), so a default `sample>=5` request is unaffected and
only genuinely oversized outputs are rejected. Cell counting uses the
selected subset's dimension SIZES (metadata only) — never the data
arrays — so a rejected request reads ZERO data chunks.
"""
from __future__ import annotations

import math
import os
from typing import Iterable, Optional

DEFAULT_MAX_BBOX_CELLS = 500_000


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


def output_cell_count(nlat: int, nlon: int, sample: int) -> int:
    """Output cell count after `sample`-stride decimation (isel
    slice(None, None, sample))."""
    if sample < 1:
        raise ValueError(f"sample must be >= 1, got {sample}")
    return math.ceil(nlat / sample) * math.ceil(nlon / sample)


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


def plan_bbox(adapter, lon0: float, lon1: float, lat0: float, lat1: float,
              sample: int, constituents: Optional[Iterable] = None,
              halo: float = 0.0, max_cells: Optional[int] = None):
    """Plan a bbox map query end-to-end up to (but not including)
    materialization. Returns (subsampled_lazy_subset, output_cells).

    Steps: select the bbox lazily (dateline-aware, adapter); read its
    dimension SIZES (metadata, no data read); compute post-sample output
    cells; enforce the cap (raise BboxCapError BEFORE any subsample /
    `.values`). On rejection NO data chunk is read."""
    if max_cells is None:
        max_cells = get_max_bbox_cells()
    sub = adapter.sel_bbox(lon0, lon1, lat0, lat1,
                           constituents=constituents, halo=halo)
    nlat, nlon = adapter.grid_shape(sub)              # dimension sizes only
    cells = output_cell_count(nlat, nlon, sample)
    if cells > max_cells:
        raise BboxCapError(cells, max_cells, nlat, nlon, sample)
    return adapter.subsample(sub, sample), cells
