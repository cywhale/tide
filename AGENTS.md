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
`conf/start_app.sh` is the VM24 production launcher: it requires the VM24
Dask TLS identity files and binds `127.0.0.1:8040`, so it exits early on
other hosts. For a local production-like stack, run Gunicorn directly. The
TPXO10 conversion environment is separate:
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
(env `TIDE_ZARR_PATH`, default `data/tpxo10.zarr`). `TIDE_ZARR_PATH` is read
only from the process environment (no `.env` loading) and a relative value
resolves against the process cwd, so operators must give an absolute store
path. Rollback to the legacy store keeps the current runtime and only
switches `TIDE_ZARR_PATH` (see `dev_tpxo10/DEPLOYMENT_VM24.md`); the adapter
detects either schema fail-closed.

Runtime code must read the store through the adapter + query planner, never
through hardcoded `xr.open_zarr(...)`. Bbox/map queries must go through
`plan_bbox()` so `MAX_BBOX_CELLS=500_000` (env
`TIDE_MAX_BBOX_CELLS`) is enforced before materialization. TPXO10 invalid
cells (`flag==2`) must propagate as missing/null, never as fabricated zero.
`TIDE_DASK_DISABLE=1` skips the Dask client for tests/single-process runs;
TPXO10 store reads use the direct adapter path, so Dask is optional and must
not be treated as part of the bbox critical path.

The shared Dask scheduler is mutual-TLS only and loopback-only (production
address `tls://127.0.0.1:8786`). The client must never connect over
plaintext or to a routable host: unless `DASK_SCHEDULER_ADDRESS` is `tls://`
with a loopback IP literal (hostnames, even `localhost`, are rejected) and
`DASK_TLS_CA_FILE`, `DASK_TLS_CLIENT_CERT`, and `DASK_TLS_CLIENT_KEY` are all
set, `src/dask_client_manager.py` creates no client and the API keeps
serving. Keep `tests/test_dask_client_manager.py` covering every refusal
case. The Tide API never submits Dask tasks; the client only holds a
scheduler connection, and removing it is a planned follow-up slice. Dask
client credentials are separate from the HTTPS
`conf/privkey.pem`/`fullchain.pem`; never commit either.

## Production Notes
VM24 production serves TPXO10 from the non-git release directory
`~/python/tide_tpxo10_atlas_v2`, not from the legacy `~/python/tide`
checkout. The legacy checkout still holds the TPXO9 store used for rollback,
but its old code is not TLS-compliant and is not a standard rollback path.
Operator details, rollback commands, and their verification status live in
`dev_tpxo10/DEPLOYMENT_VM24.md`; host-level Dask TLS operations live in the
VM24 runbook `/home/odbadmin/python/conf/DASK_TLS_RUNBOOK.md`.

v0.3.2 is a pending release: VM24 still runs v0.3.1 plus the 2026-10-24
Dask TLS hot patch. Release order: merge, then py311 tests and canary from a
separate checkout on VM24, then a production deployment only with the
owner's separate authorization, then the `v0.3.2` tag. Never modify the
release directory before the canary passes.

For deployment scripts, do not assume an interactive shell PATH.
`conf/start_app.sh` runs only an explicitly set absolute `GUNICORN_BIN` or
the VM24 py311 Gunicorn, and exits with an error otherwise. There is no
`uv run` or PATH fallback, so a PM2 start can never create or sync a uv
environment; `tests/test_start_app_launcher.py` enforces this. Public API
users cannot select TPXO9; TPXO9 is deprecated and retained only for
operator rollback.

Never run a test or canary instance through PM2 on VM24: the `tide` entry's
`pre_stop` kills every process whose command line contains `tide_app`. Start
a canary from a separate checkout or worktree, never by writing new code
into the release directory (production runs with `--reload`, so any edit
there reloads the live workers). Run it with the py311 Gunicorn on
`127.0.0.1:8041`, without PM2, `conf/start_app.sh`, or `--reload`, with an
absolute `TIDE_ZARR_PATH`, and stop it by PID. Smoke HTTPS against the
certificate's hostname with `curl --resolve <host>:8041:127.0.0.1`; never
disable verification with `curl -k`. Report Dask transport evidence
(scheduler client registration, a trivial task with the same credentials)
separately from the Tide API/store smoke. PM2 environment tests must use a
separate `PM2_HOME` on every `pm2` command, with a dummy app that has no
`pre_stop` and is not a `tide_app` process; never use the production
`conf/ecosystem.config.js` for them. The full procedures are in
`dev_tpxo10/DEPLOYMENT_VM24.md`.
