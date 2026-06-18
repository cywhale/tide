# Repository Guidelines

## Project Structure & Module Organization
Application code lives in `src/` with runtime store access centralized in
`src/store_adapter.py` and bbox planning/cap enforcement in
`src/query_planner.py`. HTTP entry points are collected in `tide_app.py`,
while deployment scripts and PM2 config sit in `conf/` (`start_app.sh`,
`ecosystem.config.js`). Model assets and Zarr grids belong in `data/`
(never commit large stores). Legacy exploratory notebooks are under `dev/`
and `examples/`; TPXO10 conversion, gate scripts, benchmark evidence, and
visual QA notebooks live under `dev_tpxo10/`. Specs for public contracts
live in `specs/`, and regression fixtures/scripts reside in `test/` and
`tests/`.

## Build, Test, and Development Commands
Use the production/runtime environment from the repo root:
```bash
uv sync
TIDE_DASK_DISABLE=1 uv run pytest
```
Run the API locally with hot reload when you do not need TLS:
```bash
uvicorn tide_app:app --reload --port 8040
```
For a production-like stack with Dask scheduler/workers, certificates, and
Gunicorn workers, use `bash conf/start_app.sh`. The TPXO10 conversion
environment is separate:
```bash
cd dev_tpxo10
uv sync
uv run pytest
```
Do not mix pyTMD generations in one environment: runtime remains on pyTMD
2.2.8, while `dev_tpxo10/` conversion/golden tools use pyTMD 3.x.

## Coding Style & Naming Conventions
Follow PEP 8: 4-space indentation, `snake_case` for functions/variables, and CapWords for Pydantic models. Keep functions vectorized with NumPy/xarray, avoid side effects in `src/` helpers, and prefer explicit type hints on public interfaces (see `tide_app.py`). Use docstrings to describe coordinate assumptions and units (e.g., meters vs. centimeters). Configuration toggles belong in `src/config.py`; avoid scattering constants in handlers.

## Testing Guidelines
Use `tests/` for pytest suites and `test/` for legacy fixtures/notebooks.
Keep synthetic station fixtures in JSON under `test/` and reuse them to
avoid touching production data. When tests depend on large Zarr archives,
gate them with environment checks so CI can skip when datasets are absent.
The v0.3.0 gate record is in `dev_tpxo10/TESTING.md` and
`dev_tpxo10/STAGE3_G3_CLOSEOUT.md`; update those files when adding new
gate evidence. `dev_tpxo10/plot_tide.ipynb` is manual visual QA for tidal
current maps, not a binding gate.

## Commit & Pull Request Guidelines
Recent history favors concise, imperative summaries (`fix truncate mode when scaling heights`). Use the first line ≤72 chars, optionally add context after a colon, and reference issues with `#id` when relevant. PRs must explain the user-facing effect, list new commands or configs, and include screenshots or API responses for UI/API tweaks. Mention data/schema migrations explicitly and confirm `pytest` plus the relevant notebooks (if any) were exercised.

## Security & Configuration Tips
Store TLS keys in `conf/` only for local testing; never check real
certificates, `.env`, API tokens, or VM credentials into Git. The API loads
the tide store at startup through the v0.3.0 store adapter
(`src/store_adapter.py`, D9): the path is resolved by `get_zarr_path()`
(env `TIDE_ZARR_PATH`, default `data/tpxo10.zarr`). Rollback to the legacy
store is a single env change `TIDE_ZARR_PATH=data/tpxo9.zarr` + restart;
the adapter detects either schema fail-closed.

Runtime code must read the store through the adapter + query planner, never
through hardcoded `xr.open_zarr(...)`. Bbox/map queries must go through
`plan_bbox()` so `MAX_BBOX_CELLS=500_000` (env
`TIDE_MAX_BBOX_CELLS`) is enforced before materialization. TPXO10 invalid
cells (`flag==2`) must propagate as missing/null, never as fabricated zero.
`TIDE_DASK_DISABLE=1` skips the Dask client for tests/single-process runs;
production keeps the shared scheduler unless deployment deliberately
changes that contract.

## v0.3.0 Deployment Boundary
G0→G3 are complete and pre-deployment evidence is tracked. Deployment is a
separate operational step: do not mutate the VM24 production repo/process
from development scripts. For rollout, use an isolated worktree, transfer
`data/tpxo10.zarr` out-of-band, keep `data/tpxo9.zarr` for rollback, and
follow `dev_tpxo10/STAGE3_G3_CLOSEOUT.md` section 5.
