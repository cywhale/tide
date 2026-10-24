# VM24 TPXO10 Deployment Notes

Initial TPXO10 cutover: 2026-06-18. Formal v0.3.1 release-directory
cleanup: 2026-07-08. Shared Dask mutual TLS: 2026-10-24 (v0.3.2).
API version: v1.1.0. Project release: v0.3.2.

## Current Production Layout

- Host: VM24 (`odb24`, `192.168.2.24`).
- PM2 process: `tide`.
- Runtime cwd: `/home/odbadmin/python/tide_tpxo10_atlas_v2` (a release copy,
  not a git checkout; detect drift by comparing file hashes with the
  released commit).
- Python runtime: VM24 pyenv `py311` (Python 3.11.4) via
  `/home/odbadmin/.pyenv/versions/py311/bin/gunicorn`. See "Runtime
  environment" below: this environment is not managed by `uv.lock`.
- HTTPS bind: `127.0.0.1:8040`.
- Runtime store: `data/tpxo10.zarr` (TPXO10-atlas-v2).
- Rollback store: `/home/odbadmin/python/tide/data/tpxo9.zarr`
  (TPXO9-atlas-v5), served by the current runtime (see "Rollback").
- Shared Dask: `tls://127.0.0.1:8786`, Tide client identity `tide-client`.

The legacy `~/python/tide` checkout (pre-v0.3.0 code) is kept unchanged
because it holds the TPXO9 store. Its code is not a standard rollback path
(see "Legacy checkout"). Public users do not select TPXO9; it is deprecated
for public API use and retained only for operator rollback.

## Cutover Summary

The release was first staged and smoked on a separate local port. The
canonical TPXO10 store was copied into the release directory, then PM2 was
switched from the old cwd to the TPXO10 release cwd. Smoke checks passed:

- `/api/tide` point time series: 200.
- `/api/tide` small bbox map: 200.
- `/api/tide` 45-degree `sample=1`: 400 with `MAX_BBOX_CELLS`.
- `/api/tide/const` multipoint: 200.

Rollback drill passed (2026-06-18): PM2 was switched back to `~/python/tide`,
a TPXO9 point query returned 200, and PM2 was then switched forward to the
TPXO10 release again. This drill predates the Dask TLS change and is no
longer representative; the current rollback procedure is below.

The temporary release directory was later renamed to the formal runtime cwd
`~/python/tide_tpxo10_atlas_v2`; PM2 was restarted from that cwd and saved.

## Shared Dask TLS (v0.3.2)

The shared scheduler and worker are mutual-TLS only and bind loopback. Host
operations (certificate issuance/renewal, scheduler/worker restarts, PM2
dump handling) are documented in the VM24 runbook
`/home/odbadmin/python/conf/DASK_TLS_RUNBOOK.md`, not in this repository.

- `conf/start_app.sh` exports `DASK_SCHEDULER_ADDRESS=tls://127.0.0.1:8786`
  and the `tide-client` CA/certificate/key paths under
  `/home/odbadmin/.config/dask/tls`, and exits if any of them is unreadable.
  Because PM2 re-runs the launcher on every restart and resurrect, the TLS
  settings do not depend on the PM2 saved environment.
- `src/dask_client_manager.py` connects only when the address is `tls://`
  with a loopback IP literal (`127.0.0.0/8` or `::1`; hostnames, even
  `localhost`, are rejected) and all three credential paths are set, and
  always passes `require_encryption=True`. Missing, partial, or mismatched
  settings create no client; the API keeps serving because TPXO10 reads use
  the direct adapter path (`chunks=None`). It never falls back to plaintext
  `tcp://` or connects to a routable scheduler.
- The Tide API itself never submits Dask tasks; it only holds a client
  connection to the scheduler.
- Dask client credentials are separate from the HTTPS
  `conf/privkey.pem`/`fullchain.pem`. Neither is committed to the repository.

## Rollback

Status: **unverified** with the v0.3.2 runtime. Mark it verified here only
after the canary procedure below passes and the PM2 environment
propagation/resurrect behavior is confirmed.

Rollback keeps the current release runtime and switches only the store, by
absolute path in both directions. `TIDE_ZARR_PATH` is read from the process
environment only (no `.env` loading), and a relative value would resolve
against the release cwd, which has no TPXO9 store. Do not rely on unsetting
the variable to switch forward: PM2 can keep an environment captured at an
earlier start, so always set the target store explicitly.

```bash
PM2=$HOME/.npm-global/bin/pm2
cd /home/odbadmin/python/tide_tpxo10_atlas_v2

# Roll back to TPXO9 (pre_stop of `tide` kills every tide_app process).
$PM2 delete tide
TIDE_ZARR_PATH=/home/odbadmin/python/tide/data/tpxo9.zarr \
  $PM2 start conf/ecosystem.config.js
$PM2 env <tide-id> | grep TIDE_ZARR_PATH    # id from `$PM2 status`

# Switch forward to TPXO10.
$PM2 delete tide
TIDE_ZARR_PATH=/home/odbadmin/python/tide_tpxo10_atlas_v2/data/tpxo10.zarr \
  $PM2 start conf/ecosystem.config.js
$PM2 env <tide-id> | grep TIDE_ZARR_PATH
```

To make either state survive a reboot, the PM2 dump must be updated. `pm2
save` rewrites the whole process list; follow the host runbook (compare
current and saved states first, or update only the `tide` entry from a
backup) instead of saving blindly.

## Legacy checkout

`~/python/tide` is not part of the standard rollback. Its pre-v0.3.0 code
creates a Dask client at import with the default plaintext address
`tcp://localhost:8786` (refused by the TLS scheduler) and opens the store
with `chunks='auto'`, so computation falls back to local threads. It is
unverified since the TLS change and does not comply with the TLS rule. Use
it only as an emergency measure after confirming it cannot reach the
scheduler, and test it first.

## Canary procedure

Use a canary to verify a release or the rollback store without touching the
production process.

Constraints:

- Start it from a separate checkout or worktree of the commit under test
  (for example under `~/tmp/`), never by writing new code into the release
  directory: production runs with `--reload`, so any edit there reloads the
  live workers.
- Never start it through PM2: the `tide` entry's `pre_stop` kills every
  process whose command line contains `tide_app`, which would hit both the
  canary and production. Do not use `conf/start_app.sh` either: it binds
  `127.0.0.1:8040` and enables `--reload`.
- The canary checkout has no `data/` stores; select the store under test
  with an absolute `TIDE_ZARR_PATH` (the TPXO9 or TPXO10 path above).
  Reference the HTTPS key/cert of the release directory by absolute path;
  never copy them into the canary checkout.

Start and stop:

- Export the same Dask TLS variables as `conf/start_app.sh` plus the
  absolute `TIDE_ZARR_PATH`, then run
  `/home/odbadmin/.pyenv/versions/py311/bin/gunicorn tide_app:app -w 1
  -k uvicorn.workers.UvicornWorker -b 127.0.0.1:8041 --timeout 180` with the
  HTTPS key/cert options and without `--reload`. Record the PID; stop the
  canary with `kill <PID>` only.

Acceptance:

1. Before starting, record the production `tide` PID and restart count
   (`pm2 jlist`), and, with a one-off client that uses the same exported TLS
   variables, the scheduler's client list:
   `client.run_on_scheduler(lambda dask_scheduler: sorted(dask_scheduler.clients))`.
   Production holds one `Client-tideapi-*` entry per Gunicorn worker.
2. While the canary runs, the list gains exactly one `Client-tideapi-*`
   entry, the scheduler log shows
   `Receive client connection: Client-tideapi-*` at the canary start time,
   and the canary log has no `Dask client connection failed` warning. This
   shows the canary's TLS settings were accepted by the scheduler.
3. The same one-off client completes a trivial task, for example
   `client.submit(lambda x: x + 1, 41).result()` returning `42`. This shows
   the same TLS settings and `tide-client` identity can reach the worker.
   The Tide API never submits tasks itself; do not read an API response as
   Dask evidence.
4. API smoke on `https://127.0.0.1:8041` passes and shows the intended store
   is served (use a query whose TPXO9 and TPXO10 answers differ).
5. After `kill <PID>`, the client list returns to the step 1 baseline, and
   the production `tide` PID and restart count are unchanged.

## PM2 environment validation

The rollback depends on how PM2 captures `TIDE_ZARR_PATH` at `pm2 start` and
whether `pm2 save`/`pm2 resurrect` preserve it. Validate this without the
production PM2 daemon and without the `tide` entry:

- Use a separate `PM2_HOME` (for example `~/tmp/pm2-envprobe`) so the
  production daemon and dump are untouched.
- Use a dummy ecosystem file whose app is not a `tide_app` process (for
  example `bash -c 'echo "$TIDE_ZARR_PATH"; sleep 3600'`) and has no
  `pre_stop`. A separate `PM2_HOME` does not sandbox commands run by
  `pre_stop`; the production entry's `pre_stop` would still kill real
  `tide_app` processes. Never use the production `conf/ecosystem.config.js`
  for this test.
- Check with `pm2 env <id>` after: a start with the TPXO9 path; a delete and
  start with the TPXO10 path; a restart from a shell where the variable is
  unset (to observe whether PM2 keeps the old value); and `pm2 save`,
  `pm2 kill`, `pm2 resurrect`. Finish with `pm2 kill` for that `PM2_HOME`.

## Runtime environment

Production runs on the shared VM24 py311 pyenv, not on an environment built
from `uv.lock`. On 2026-10-24, 12 of 18 runtime packages differed from
`uv.lock`; for example dask/distributed 2025.3.0 vs 2025.11.0, numpy 2.2.4
vs 2.3.4, xarray 2025.3.1 vs 2025.10.1, fastapi 0.115.12 vs 0.121.1. pyTMD
2.2.8, scipy, numcodecs, and netCDF4 match. Repository tests run in the
`uv.lock` environment, so changes that depend on package behavior must also
be checked in py311 (for example in a canary).

The launcher runs only the py311 Gunicorn (or an explicit absolute
`GUNICORN_BIN`), so a PM2 start can never pick `uv run`. On 2026-10-24,
before this rule, a PM2 restart whose PATH included `uv` started a `uv` sync that
created an incomplete `.venv` in the release directory; it was moved
(not deleted) to `/home/odbadmin/tmp/tide-uv-venv-incomplete-20261024`.

## Operational Notes

- `conf/start_app.sh` runs only an explicitly set absolute `GUNICORN_BIN` or
  the VM24 py311 Gunicorn, and exits with code 127 if neither is usable.
  There is no `uv run` or PATH fallback.
- The launcher runs Gunicorn with `--reload`, so editing Python sources in
  the release directory reloads the workers immediately.
- TPXO10 coastline reclassification can return no value where TPXO9 used to
  serve extrapolated values.
- Large bbox maps are capped by `MAX_BBOX_CELLS=500000` after applying
  `sample`.
- A non-blocking Polars CPU feature warning may appear in PM2 logs on VM24.

## Open Operational Items

- Decide whether Tide needs a Dask client at all. No Tide runtime path uses
  it (reads are direct); removing it would also allow retiring the
  `tide-client` identity.
- Make the py311 runtime reproducible (a lock or recorded freeze) or move
  production to an environment built from `uv.lock`.
- Parameterize host-specific paths in `conf/start_app.sh` (the launcher
  tests currently rewrite them in a sandbox copy).
- Remove `--reload` from the production launcher.
- Narrow the `tide` entry's `pre_stop` process match.
- Clean up merged branches and VM24 temporary directories once the above is
  settled.
