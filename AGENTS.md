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
uv run pytest
```
Run the API locally with hot reload when you do not need TLS:
```bash
uvicorn tide_app:app --reload --port 8040
```
`conf/start_app.sh` is the VM24 production launcher: it selects the approved
Gunicorn binary and binds `127.0.0.1:8040`; Tide does not need Dask TLS files.
For a local production-like stack, run Gunicorn directly. Dask is kept in the
`dask-tools` dependency group for local simulation and benchmark tools:
```bash
uv sync --group dask-tools
```
The TPXO10 conversion environment is separate:
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
TPXO10 store reads use the direct adapter path (`chunks=None`). The Tide API
does not create a Dask client or submit Dask tasks. Dask remains available in
the separate `dask-tools` dependency group for benchmarks and local
simulation; it is not part of the Tide API runtime dependency set.

The shared Dask scheduler remains mutual-TLS and loopback-only for services
that use it, including MHW. This Tide change removes only the Tide API's
client and its `tide-client` credential requirement; it does not stop or
reconfigure the scheduler, worker, MHW client, or their credentials. The
`src/dask_client_manager.py` module is a no-op compatibility shim during the
runtime rollout. Never commit Dask or HTTPS credentials.

## Production Notes
VM24 production serves TPXO10 from the non-git release directory
`~/python/tide_tpxo10_atlas_v2`, not from the legacy `~/python/tide`
checkout. The legacy checkout still holds the TPXO9 store used for rollback,
but its old code is not TLS-compliant and is not a standard rollback path.
Operator details, rollback commands, and their verification status live in
`dev_tpxo10/DEPLOYMENT_VM24.md`; host-level Dask TLS operations live in the
VM24 runbook `/home/odbadmin/python/conf/DASK_TLS_RUNBOOK.md`.

v0.3.2 was deployed and tagged `v0.3.2` on 2026-11-03. The separate Tide
Dask-client-removal change in this branch is not yet deployed; production may
still use the v0.3.2 client until a separately authorized rollout. Never
modify the release directory before the applicable canary passes. A
deployment writes only the
runtime manifest files, each staged outside the release directory, verified
by hash, and swapped in with its own atomic rename (Gunicorn runs with
`--reload`); use `conf/replace_runtime_files.sh` (`deploy`/`restore`) rather
than hand-typed commands. The renames together are not one transaction: if
`deploy` fails after the first one, run `restore` before anything else.
That script and `conf/https_smoke.sh` are fail-fast and covered by
`tests/test_ops_scripts.py`. Run both from the merged checkout
(they are not deployed). A deployment is not complete until
`conf/pm2_restart_check.sh --pm2-home /home/odbadmin/.pm2` has restarted
`tide` with the stored environment (no `--update-env`) and printed
`OK restart`, and the production smoke chained after it with `&&` has
passed, because launcher changes apply only on restart. The restart check
ignores an inherited `PM2_HOME`; always name the target daemon with
`--pm2-home`. If it reports "no running PM2 daemon" or "died or changed
during the run", stop and report to the owner. Run no `pm2` command at all
(a `pm2` CLI that cannot reach a daemon starts one), and do not run
`restore` or rerun the restart and smoke on your own: `restore` does not
call `pm2`, but it rewrites release files and can trigger a Gunicorn
`--reload`. Inspect read-only with `pm2_restart_check.sh --daemon-only`,
which never calls `pm2`, and `ps`. Any recovery, including `restore` and
`pm2 resurrect`, follows the owner's decision. `https_smoke.sh --out-root`
gives every attempt its own directory. Rollback status is "components
validated" at best until an owner-authorized production rollback drill is
run.

Never `pip install` into the shared VM24 py311 by hand. It also serves MHW
and the Dask scheduler and worker, and Tide and MHW run with `--reload`, so
they reload as soon as package files change. The owner-authorized Zarr
2.18.7 alignment was applied to the shared VM24 py311 on 2026-11-03 through
`conf/py311_zarr_align.sh`. The validated stop-first order stopped MHW, Tide,
the Dask worker, and the Dask scheduler with
`conf/pm2_app_lifecycle.sh`, required no process to use py311
(`conf/py311_quiescent.sh`, run as root, fails closed on any live process it
cannot inspect), applied Zarr, verified the result with
`conf/py311_zarr_state.sh`, and started the services with their checks.
Services never started against `STATE unknown`. Start a stopped PM2 app with `pm2 restart`
(as the lifecycle script does), never `pm2 start NAME`, which can create a
new app. This 2026-11-03 maintenance is complete; do not repeat it without a
new owner-approved change. Do not assume the MHW cron uses py311.

For deployment scripts, do not assume an interactive shell PATH.
`conf/start_app.sh` runs only an explicitly set absolute `GUNICORN_BIN` or
the VM24 py311 Gunicorn, and exits with an error otherwise. There is no
`uv run` or PATH fallback, so a PM2 start can never create or sync a uv
environment; `tests/test_start_app_launcher.py` enforces this. Public API
users cannot select TPXO9; TPXO9 is deprecated and retained only for
operator rollback.

Never run a test or canary instance through PM2 on VM24: the `tide` entry is
the production app, and its `pre_stop` would kill every process whose command
line contains `tide_app` (PM2 5.4.2 never runs `pre_stop`; do not rely on it
either way). Start
a canary from a separate checkout or worktree, never by writing new code
into the release directory (production runs with `--reload`, so any edit
there reloads the live workers). Run it with the py311 Gunicorn on
`127.0.0.1:8041`, without PM2, `conf/start_app.sh`, or `--reload`, with an
absolute `TIDE_ZARR_PATH`, and stop it by PID. Smoke HTTPS against the
certificate's hostname with `conf/https_smoke.sh` (`curl --resolve`,
`--fail-with-body`, HTTP 200 required); never disable verification with
`curl -k`. Compare against production through the public HTTPS frontend
with an explicit `--prod-port 443`. The internal Gunicorn listener on 8040
still serves an expired certificate, so verified requests to it fail. Use
a valid certificate for the canary itself, not the release directory's
`conf/*.pem`. Dask scheduler TLS checks apply to the shared cluster and its
clients, not Tide API startup after the client-removal change. PM2 dummy
configs under `/home/odbadmin/tmp` must be `.cjs`,
because that tree is in ES module scope. Report Dask transport evidence
(scheduler client registration, a trivial task with the same credentials)
separately from the Tide API/store smoke only when testing the shared Dask
service itself; the Tide API no longer has a Dask client in this branch.
PM2 environment tests must use a
separate `PM2_HOME` on every `pm2` command, with a dummy app that has no
`pre_stop` and is not a `tide_app` process; never use the production
`conf/ecosystem.config.js` for them. The full procedures are in
`dev_tpxo10/DEPLOYMENT_VM24.md`.
