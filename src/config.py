dz = None
gridSz = None
timeLimit = None
LON_RANGE_LIMIT = None
LAT_RANGE_LIMIT = None
AREA_LIMIT = None
cons = None #['q1','o1','p1','k1','n2','m2','s1','s2','k2','m4','ms4','mn4','2n2','mf','mm']

# v0.3.0 Stage 3: store path + schema adapter live here so every read path
# resolves the store through one helper (rollback = single env var change).
adapter = None  # set at lifespan startup to a store_adapter.StoreAdapter

# MAX_BBOX_CELLS = 500_000 (spec §7.5.3, owner+reviewer signed 2026-06-15):
# protective cap counted on the POST-sample output grid, enforced before
# any .compute()/.values/prediction; env-overrideable (TIDE_MAX_BBOX_CELLS).
# Resolved once at lifespan startup via query_planner.get_max_bbox_cells().
MAX_BBOX_CELLS = None

# re-exported so the spec's `config.get_zarr_path()` resolves; the
# implementation lives in store_adapter to avoid a config<-adapter cycle.
from src.store_adapter import get_zarr_path  # noqa: E402,F401
from src.query_planner import get_max_bbox_cells  # noqa: E402,F401