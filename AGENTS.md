# Repository Guidelines

## Project Structure & Module Organization
Application code lives in `src/` with `model_extract.py`, `model_utils.py`, and FastAPI helpers referenced by `tide_app.py`. HTTP entry points are collected in `tide_app.py`, while deployment scripts and PM2 config sit in `conf/` (`start_app.sh`, `ecosystem.config.js`). Model assets and Zarr grids belong in `data/` (never commit large updates), and exploratory notebooks are under `dev/` and `examples/`. Specs for public contracts live in `specs/`, and regression notebooks/scripts reside in `test/`.

## Build, Test, and Development Commands
Create a virtual environment and install the package editable so the `src` module resolves:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt
```
Run the API locally with hot reload when you don’t need TLS:
```bash
uvicorn tide_app:app --reload --port 8040
```
For a production-like stack with Dask scheduler/workers, certificates, and Gunicorn workers, use `bash conf/start_app.sh`.
Execute the regression notebooks/tests with `python -m pytest test/` or target a single scenario (e.g., `pytest test/simu_tide_resp01.py`) before opening a PR.

## Coding Style & Naming Conventions
Follow PEP 8: 4-space indentation, `snake_case` for functions/variables, and CapWords for Pydantic models. Keep functions vectorized with NumPy/xarray, avoid side effects in `src/` helpers, and prefer explicit type hints on public interfaces (see `tide_app.py`). Use docstrings to describe coordinate assumptions and units (e.g., meters vs. centimeters). Configuration toggles belong in `src/config.py`; avoid scattering constants in handlers.

## Testing Guidelines
Pytest discovers cases in `test/`; name new files `test/<purpose>_<scenario>.py` to match the existing pattern. Keep synthetic station fixtures in JSON under `test/` and reuse them to avoid touching production data. When tests depend on the Zarr archive, gate them with `pytest.mark.slow` or environment checks so CI can skip when datasets are absent. Targeted plots (PNG artifacts) should be generated into `test/` and git-ignored unless they demonstrate a regression fix.

## Commit & Pull Request Guidelines
Recent history favors concise, imperative summaries (`fix truncate mode when scaling heights`). Use the first line ≤72 chars, optionally add context after a colon, and reference issues with `#id` when relevant. PRs must explain the user-facing effect, list new commands or configs, and include screenshots or API responses for UI/API tweaks. Mention data/schema migrations explicitly and confirm `pytest` plus the relevant notebooks (if any) were exercised.

## Security & Configuration Tips
Store TLS keys in `conf/` only for local testing—never check real certificates into Git. The API loads the tide store at startup through the v0.3.0 store adapter (`src/store_adapter.py`, D9): the path is resolved by `get_zarr_path()` (env `TIDE_ZARR_PATH`, default `data/tpxo10.zarr`); a rollback to the legacy store is a single env change `TIDE_ZARR_PATH=data/tpxo9.zarr` + restart (the adapter detects the schema fail-closed and serves either). Runtime code must read the store through the adapter + query planner (`src/query_planner.py`), never via a hardcoded `xr.open_zarr(...)`. Bbox/map queries must go through `plan_bbox()` so the `MAX_BBOX_CELLS=500_000` cap (env `TIDE_MAX_BBOX_CELLS`) is enforced before any materialization. Guard new code with informative errors when the dataset is missing. When adjusting Dask memory limits or grid sizes, expose them through `src/config.py` so operations and agents can override via environment variables; `TIDE_DASK_DISABLE=1` skips the Dask client (tests / single-process).
