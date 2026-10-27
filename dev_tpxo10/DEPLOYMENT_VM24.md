# VM24 TPXO10 Deployment Notes

Initial TPXO10 cutover: 2026-06-18. Formal v0.3.1 release-directory
cleanup: 2026-07-08. Shared Dask mutual TLS on VM24: 2026-10-24 (host change
plus a hot patch of two Tide files). API version: v1.1.0.

- Released (tagged): v0.3.1.
- Target/pending release: v0.3.2. Not tagged and not deployed; see
  "v0.3.2 release sequence".

## Current Production Layout

- Host: VM24 (`odb24`, `192.168.2.24`).
- PM2 process: `tide`.
- Runtime cwd: `/home/odbadmin/python/tide_tpxo10_atlas_v2` (a release copy,
  not a git checkout). Detect drift by comparing the hashes of the runtime
  files (`tide_app.py`, `__init__.py`, `src/*.py`, `conf/start_app.sh`,
  `conf/ecosystem.config.js`) with the deployed commit. Docs, tests, and
  notebooks are not deployed and may lag.
- Code: v0.3.1 plus the 2026-10-24 Dask TLS hot patch of `conf/start_app.sh`
  and `src/dask_client_manager.py`. The hot-patched files are recorded
  verbatim in the repository as the first commit of PR #2. The hardened
  v0.3.2 versions are not deployed yet.
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

## Shared Dask TLS

The shared scheduler and worker are mutual-TLS only and bind loopback. Host
operations (certificate issuance/renewal, scheduler/worker restarts, PM2
dump handling) are documented in the VM24 runbook
`/home/odbadmin/python/conf/DASK_TLS_RUNBOOK.md`, not in this repository.

Tide keeps its Dask client in v0.3.2; this release only hardens the client's
TLS behavior. The Tide API never submits Dask tasks: reads use the direct
adapter path (`chunks=None`), and the client only holds a connection to the
scheduler. Removing the client is a separate follow-up slice.

What VM24 runs now (2026-10-24 hot patch):

- `conf/start_app.sh` exports `DASK_SCHEDULER_ADDRESS=tls://127.0.0.1:8786`
  and the `tide-client` CA/certificate/key paths under
  `/home/odbadmin/.config/dask/tls`, and exits if any of them is unreadable.
  It prefers the py311 Gunicorn but still falls back to `uv run` and PATH
  when py311 is missing.
- `src/dask_client_manager.py` passes `Security(...,
  require_encryption=True)` when all three credential paths are set. With
  none set it would still try plaintext `tcp://localhost:8786`; the launcher
  always sets them, so this path is not taken on VM24.

v0.3.2 target (this repository, pending deployment):

- The client connects only when the address is `tls://` with a loopback IP
  literal (`127.0.0.0/8` or `::1`; hostnames, even `localhost`, are
  rejected) and all three credential paths are set, and always passes
  `require_encryption=True`. Missing, partial, or mismatched settings create
  no client and never fall back to plaintext `tcp://` or a routable
  scheduler; the API keeps serving.
- The launcher keeps the same TLS exports and runs only an explicitly set
  absolute `GUNICORN_BIN` or the VM24 py311 Gunicorn, exiting with code 127
  otherwise. The `uv run` and PATH fallbacks are removed. `--reload` is
  unchanged.

The launcher re-runs on every PM2 restart and resurrect, so the TLS settings
do not depend on the PM2 saved environment. Dask client credentials are
separate from the HTTPS `conf/privkey.pem`/`fullchain.pem`. Neither is
committed to the repository.

## v0.3.2 release sequence

The production release directory must not be modified until step 5. Its
Gunicorn runs with `--reload`, so any file written there reloads the live
workers.

1. PR #2 is reviewed and the owner merges it into `main`. No tag; no
   production change.
2. Codex creates a separate checkout or worktree of the merged commit on
   VM24 (for example `~/tmp/tide-v0.3.2-<sha>`). Link `data/tpxo10.zarr` and
   `data/tpxo9.zarr` there to the existing stores as read-only symlinks
   (never copy them). Run the test suite with the VM24 py311 interpreter,
   for example
   `TIDE_DASK_DISABLE=1 /home/odbadmin/.pyenv/versions/py311/bin/python -m pytest -rs`,
   and record the passed, skipped, and warning counts. If pytest is not
   installed in py311, do not install it into that shared environment; use
   a throwaway venv created from py311 with `--system-site-packages` and
   install pytest only there. Local `uv.lock` runs do not count as py311
   verification.
3. From the same checkout, Codex runs the canary (below) against both
   stores, TPXO10 for the target runtime and TPXO9 for rollback, and the PM2
   environment validation (below). Neither touches the release directory or
   the production PM2 daemon.
4. The owner gives separate authorization for the production update.
5. **Production deployment.** This changes the live service. Deploy only
   the v0.3.2 runtime manifest by atomic replacement, then restart `tide`
   and run the post-deploy checks, as described in "Production deployment
   procedure" below.
6. Update this file with the outcome: rollback status "components
   validated" if the TPXO9 canary and the PM2 validation passed (see
   "Rollback"). Tag `v0.3.2` only after the validation and the production
   deployment are complete.

### Outstanding rollout checks

Each item stays open until it has been done once with its result (date,
counts, outputs) recorded here.

- [ ] py311 test suite in a separate VM24 checkout (step 2): passed,
      skipped, and warning counts.
- [ ] Before the canary: certificate SAN hostname (canary part B, step 1)
      and VM24 `curl --version` (7.76 or later) recorded.
- [ ] `conf/replace_runtime_files.sh` rehearsed on a scratch copy (see
      "Production deployment procedure").
- [ ] Canary parts A, B, and C passed for the TPXO10 and TPXO9 stores.
- [ ] PM2 environment validation passed with an isolated `PM2_HOME`,
      including the `pm2_restart_check.sh` rehearsal on the dummy app.
- [ ] Owner authorization for the production deployment (step 4).

## Production deployment procedure

Run this only at release sequence step 5, with the owner's authorization.

Deployment manifest for v0.3.2 (runtime files only):

- `conf/start_app.sh`
- `src/dask_client_manager.py`

Docs, tests, and notebooks are not deployed. They have no runtime role, and
each extra write into the release directory is avoidable risk.

Gunicorn runs with `--reload`, so it can reload while a file is being
written. Never copy over a runtime file in place; use
`conf/replace_runtime_files.sh` from the merged-commit checkout. It runs
under `set -euo pipefail`, exits non-zero on any failure, and:

- refuses a `--work` directory that already exists, so earlier backups are
  never overwritten;
- checks that the checkout is at `--commit` and that `--work` is on the
  release filesystem (`rename(2)` is atomic only within one filesystem);
- backs up the manifest files with their hashes (`backup/`,
  `backup.sha256`) and records all runtime hashes
  (`runtime-before.sha256`);
- stages every manifest file and verifies it against its blob at the commit
  before replacing any, so a mismatch replaces nothing; then swaps each in
  with one rename and re-verifies it;
- confirms that the other runtime files are unchanged.

`tests/test_ops_scripts.py` exercises these refusals, the two-phase
behavior, the rename, recovery by `restore` after a failure between the two
renames, and `restore`'s record checks locally.

```bash
REL=/home/odbadmin/python/tide_tpxo10_atlas_v2
SRC=/home/odbadmin/tmp/tide-v0.3.2-$SHA                  # merged-commit checkout
WORK=/home/odbadmin/tmp/tide-v0.3.2-deploy-$DEPLOY_DATE  # must not exist yet
"$SRC/conf/replace_runtime_files.sh" deploy --rel "$REL" --src "$SRC" \
  --commit "$SHA" --work "$WORK"
```

`src/dask_client_manager.py` is replaced last and triggers one `--reload`
of the live workers. The launcher change takes effect only when the PM2
process restarts, so the restart under "Restart and post-deploy checks" is a
required part of the same authorized deployment. Each rename is atomic, but
the two together are not one transaction: if `deploy` fails between them,
run `restore` (below) before anything else.

To back out, including after a deploy that failed partway (for example
after the first rename), `restore` puts back the backups from that `WORK`
directory with the same staging, verification, and rename. It first
requires `backup.sha256` to list exactly the two manifest paths, once each
and well formed, and refuses a truncated, duplicated, or altered record
before touching any file.

Like `deploy`, `restore` is a production change. It rewrites runtime files
in the live release directory, and replacing `src/dask_client_manager.py`
can trigger a Gunicorn `--reload` of the live workers. It runs only within
the authorized deployment, or later on the owner's decision:

```bash
"$SRC/conf/replace_runtime_files.sh" restore --rel "$REL" --work "$WORK"
```

Rehearse before step 5 on VM24, with `REL` set to a scratch copy of the
release directory's runtime files (the drift-detection list above) under
`/home/odbadmin/tmp/`:

1. `deploy` succeeds and the files match the commit.
2. Re-running `deploy` with the same `WORK` is refused.
3. `deploy` from a scratch clone whose manifest file was edited fails with
   "nothing was replaced", and the scratch copy is unchanged.
4. `restore` puts back the original files.

### Restart and post-deploy checks

Run this only after `deploy` printed `OK deploy`, in the same shell, so
that `SRC` and `SHA` are still set. Both scripts come from the merged
checkout; neither is part of the runtime manifest, so neither exists in the
release directory.

`conf/pm2_restart_check.sh` runs `pm2 restart tide` with the stored
environment. It never passes `--update-env`, which would copy this shell's
variables (for example `SHA` or a stray `TIDE_ZARR_PATH`) into the
production process.

It always acts on the daemon given by `--pm2-home`, which defaults to
production `/home/odbadmin/.pm2` and is passed explicitly below. An
inherited `PM2_HOME` is ignored, so a shell left pointing at the isolated
test daemon cannot redirect the restart.

Before any `pm2` call, it proves the daemon is alive without running `pm2`:

- `rpc.sock` is a socket;
- the PID in `pm2.pid` is running;
- that process's command line is the PM2 God Daemon title for this exact
  home (`PM2 vX: God Daemon (<PM2_HOME>)`);
- `rpc.sock` accepts a connection.

A stale socket, a plain file, a dead or reused PID, or another home's daemon
is refused without invoking `pm2`. A `pm2` CLI that cannot reach a daemon
starts a new one, so this prevents a new daemon only for a missing or stale
daemon that exists before the run.

It cannot close a narrow race: if the daemon exits after the check but
before a `pm2` call reaches it, `pm2` may still start a new daemon. To catch
that case, the check returns the PID it verified, which becomes the
baseline. At the end the script re-checks and fails ("died or changed during
the run") unless that same PID is still the live daemon, so such a run is
never reported as `OK restart`. Handling that failure is described under
"Daemon failures" below.

After the liveness check, the script fails unless all of the following
hold:

- the restart command exits 0;
- within `--timeout` (default 60 s), `tide` is `online` with a new PID;
- the restart count rose by exactly one;
- after `--settle` (default 10 s), the PID and restart count are unchanged,
  so there is no crash loop;
- the process command line contains the py311 Gunicorn path.

The smoke is chained with `&&`, so it runs only after `OK restart`. A
smoke `PASS` without a preceding `OK restart` in the same run is not
deployment evidence. Each run of `https_smoke.sh` writes to a new unique
directory under `--out-root` (created with `mktemp -d` and printed as
`attempt:`), so a retry is never blocked, even within the same second, and
a failed attempt's output is kept as evidence.

```bash
HOST=eco.odb.ntu.edu.tw   # the SAN hostname recorded before the canary
"$SRC/conf/pm2_restart_check.sh" --pm2-home /home/odbadmin/.pm2 --name tide \
  && "$SRC/conf/https_smoke.sh" --host "$HOST" --mode prod \
       --out-root "/home/odbadmin/tmp/tide-v0.3.2-$SHA-smoke"
```

The deployment is complete only if all of the following hold:

- `deploy` printed `OK deploy`. It has already verified both manifest files
  against the commit and that every other runtime file is unchanged.
- `pm2_restart_check.sh` printed `OK restart`, and `https_smoke.sh`, in the
  same chain, printed `PASS prod`.
- The `tide` log since the restart has no `Dask client connection failed`
  warning.
- The scheduler client list (method in canary part A, step 1) shows one
  `Client-tideapi-*` entry per Gunicorn worker, which is two.

If a check fails while the daemon is alive (any failure except the two
daemon failures below), run `restore`, then run the same restart-and-smoke
chain again. The new attempt gets its own directory under `--out-root`, and
the earlier one is kept.

#### Daemon failures

This applies when the restart check fails with "no running PM2 daemon" or
"died or changed during the run". Stop and report to the owner.

- Do not run `pm2 ls`, any other `pm2` command, or `pm2 resurrect`: a `pm2`
  CLI that cannot reach a daemon starts a new one.
- Do not run `restore` or rerun the restart-and-smoke chain on your own.

Inspect read-only, without `pm2`:

```bash
"$SRC/conf/pm2_restart_check.sh" --daemon-only --pm2-home /home/odbadmin/.pm2
ps -eo pid,args | grep '[G]od Daemon'
```

`--daemon-only` runs only the liveness check, prints the live daemon's PID
or the reason it is not alive, and never calls `pm2`.

`restore` does not call the `pm2` CLI, so it cannot start a PM2 daemon by
itself. It still rewrites release files, which can trigger a Gunicorn
`--reload`, so it is not a no-op for production. If the owner decides to
roll the files back, run `restore` then, and handle the service state that
follows by the procedure the owner approves.

Any PM2 recovery is a separate procedure that needs the owner's approval
first; it is not a routine fix. That includes letting `pm2` start a
daemon, `pm2 resurrect` (it restores the entire saved process list, and
saved states can differ from what was running; see the host runbook's note
on `gateway`), and restarting other apps.

## Rollback

Status: **unverified**.

- Once the TPXO9 canary and the isolated PM2 environment validation pass,
  record "rollback components validated (<date>)". Those tests cover the
  TPXO9 data path through the current runtime and how PM2 passes
  `TIDE_ZARR_PATH`, using a canary and a dummy app.
- They do not switch the production `tide` entry, so record "production
  rollback not exercised" alongside, including after the v0.3.2
  deployment. Only an owner-authorized production drill can change that.

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
$PM2 env "$($PM2 id tide | tr -dc '0-9')" | grep TIDE_ZARR_PATH

# Switch forward to TPXO10.
$PM2 delete tide
TIDE_ZARR_PATH=/home/odbadmin/python/tide_tpxo10_atlas_v2/data/tpxo10.zarr \
  $PM2 start conf/ecosystem.config.js
$PM2 env "$($PM2 id tide | tr -dc '0-9')" | grep TIDE_ZARR_PATH
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

A canary verifies a commit or the rollback store before any production
change. It must leave the production process and the release directory
untouched.

Constraints:

- Run it from a separate checkout or worktree of the commit under test
  (release sequence step 2). Never write new code into the release directory
  for a canary.
- Never start it through PM2: the `tide` entry's `pre_stop` kills every
  process whose command line contains `tide_app`, which would hit both the
  canary and production. Do not use `conf/start_app.sh` either: it binds
  `127.0.0.1:8040` and enables `--reload`.
- Always select the store under test with an absolute `TIDE_ZARR_PATH` (the
  TPXO9 or TPXO10 path above); do not rely on `data/` in the checkout.
- Reference the release directory's HTTPS key/cert by absolute path; never
  copy them into the canary checkout.

Start and stop:

- Export the same Dask TLS variables as `conf/start_app.sh` plus the
  absolute `TIDE_ZARR_PATH`, then from the canary checkout run
  `/home/odbadmin/.pyenv/versions/py311/bin/gunicorn tide_app:app -w 1
  --worker-class uvicorn.workers.UvicornWorker -b 127.0.0.1:8041
  --timeout 180 --keyfile <release>/conf/privkey.pem
  --certfile <release>/conf/fullchain.pem`, without `--reload`. Record the
  PID; stop the canary with `kill <PID>` only.

Acceptance has three separate parts.

A. Dask transport (TLS). This shows the canary's TLS settings and the
`tide-client` identity work with the scheduler and worker. It does not test
Tide computation, because the Tide API never submits Dask tasks.

1. Before starting, use a one-off client with the same exported TLS
   variables to record the scheduler's client list:
   `client.run_on_scheduler(lambda dask_scheduler: sorted(dask_scheduler.clients))`.
   Production holds one `Client-tideapi-*` entry per Gunicorn worker.
2. While the canary runs, the list gains exactly one `Client-tideapi-*`
   entry, the scheduler log shows `Receive client connection:
   Client-tideapi-*` at the canary start time, and the canary log has no
   `Dask client connection failed` warning.
3. The same one-off client completes a trivial worker task, for example
   `client.submit(lambda x: x + 1, 41).result()` returning `42`.
4. After `kill <PID>`, the client list returns to the step 1 baseline.

B. Tide API and store. This shows the canary serves the intended store.
Verify HTTPS against the certificate's hostname instead of disabling
verification; never use `curl -k`.

1. Confirm the certificate hostname:
   `openssl x509 -in <release>/conf/fullchain.pem -noout -ext subjectAltName`.
   The commands below assume `eco.odb.ntu.edu.tw`; if the certificate lists
   a different name, use that name.
2. Run `conf/https_smoke.sh` from the canary checkout. It queries the canary
   (8041) and production (8040, a read-only GET) through
   `curl --resolve HOST:PORT:127.0.0.1` with verification on and
   `--fail-with-body`, and stops at the first failure. Every response must be
   HTTP 200 with a body that is non-empty and not `{}`. It requires curl
   7.76 or later (for `--fail-with-body`) and refuses an older curl before
   sending any request. Each run writes the bodies to a new unique
   directory under `--out-root` (outside the release directory), created
   with `mktemp -d` and printed as `attempt:`. Retries never collide, and
   failed attempts are kept as evidence. The query fixes `start` and `end`,
   so the body is deterministic.

   ```bash
   SMOKE=/home/odbadmin/tmp/tide-v0.3.2-$SHA/conf/https_smoke.sh
   ROOT=/home/odbadmin/tmp/tide-v0.3.2-$SHA-smoke
   # TPXO10 canary: body must be identical to production.
   "$SMOKE" --host eco.odb.ntu.edu.tw --mode same --out-root "$ROOT"
   # TPXO9 canary: both 200, bodies must differ.
   "$SMOKE" --host eco.odb.ntu.edu.tw --mode different --out-root "$ROOT"
   ```

C. Production unaffected. The production `tide` PID and restart count
(`pm2 jlist`) and the release directory's file hashes are unchanged from
before the canary.

## PM2 environment validation

The rollback depends on how PM2 captures `TIDE_ZARR_PATH` at `pm2 start`
and whether `pm2 save`/`pm2 resurrect` preserve it. Validate this with an
isolated PM2 daemon and a dummy app, never with the production daemon or
the `tide` entry:

- Every `pm2` command in this test must carry the isolated `PM2_HOME`. A
  `pm2` command without it acts on the production daemon; `pm2 kill`
  without it would stop every production service.
- The dummy app is not a `tide_app` process and has no `pre_stop`. A
  separate `PM2_HOME` does not sandbox commands run by `pre_stop`, so never
  use the production `conf/ecosystem.config.js` for this test.

```bash
PROBE=/home/odbadmin/tmp/pm2-envprobe
PM2=$HOME/.npm-global/bin/pm2
mkdir -p "$PROBE"
cat > "$PROBE/envprobe.config.js" <<'JS'
module.exports = { apps: [{
  name: 'envprobe',
  script: 'bash',
  args: ['-c', 'echo "TIDE_ZARR_PATH=$TIDE_ZARR_PATH"; sleep 3600'],
  autorestart: false
}] };
JS
# What PM2 stored, and what the process itself printed at its last start.
probe() {
  PM2_HOME="$PROBE" "$PM2" env "$(PM2_HOME="$PROBE" "$PM2" id envprobe | tr -dc '0-9')" | grep TIDE_ZARR_PATH
  PM2_HOME="$PROBE" "$PM2" logs envprobe --nostream --raw --lines 1
}

PM2_HOME="$PROBE" TIDE_ZARR_PATH=/home/odbadmin/python/tide/data/tpxo9.zarr \
  "$PM2" start "$PROBE/envprobe.config.js"; probe
PM2_HOME="$PROBE" "$PM2" delete envprobe
PM2_HOME="$PROBE" TIDE_ZARR_PATH=/home/odbadmin/python/tide_tpxo10_atlas_v2/data/tpxo10.zarr \
  "$PM2" start "$PROBE/envprobe.config.js"; probe
# Does PM2 keep the old value when the variable is unset in the shell?
env -u TIDE_ZARR_PATH PM2_HOME="$PROBE" "$PM2" restart envprobe --update-env; probe
PM2_HOME="$PROBE" "$PM2" save
PM2_HOME="$PROBE" "$PM2" kill
PM2_HOME="$PROBE" "$PM2" resurrect; probe
PM2_HOME="$PROBE" "$PM2" kill
```

Record each `probe` output. This part of "rollback components validated"
passes only if each explicit start shows the requested path and the
save/kill/resurrect cycle preserves it.

Before the final `pm2 kill`, rehearse the restart check on the dummy app
with the same isolated daemon. This runs `pm2_restart_check.sh` with GNU
`ps` and real PM2 without touching production. The script ignores an
inherited `PM2_HOME`, so the isolated daemon is passed with `--pm2-home`:

```bash
# The dummy's command line may start with bash, /usr/bin/bash, or sleep;
# take the executable from the running process instead of guessing.
EXE=$(ps -o args= -p "$(PM2_HOME="$PROBE" "$PM2" pid envprobe)" | awk '{print $1}')
/home/odbadmin/tmp/tide-v0.3.2-$SHA/conf/pm2_restart_check.sh \
  --pm2 "$PM2" --pm2-home "$PROBE" --name envprobe --expect-exec "$EXE" --settle 3
```

It must print `daemon: PM2_HOME=` followed by the `$PROBE` path, then
`OK restart`. Passing the liveness check also confirms that this PM2
version titles its daemon `God Daemon (<PM2_HOME>)`. Record the output.

Also confirm that the script fails when run with a wrong `--expect-exec`,
and that `--daemon-only --pm2-home "$PROBE"` reports the probe daemon's PID
while it runs. After the final `PM2_HOME="$PROBE" "$PM2" kill`, run it once
more with `--pm2-home "$PROBE"`, and once with `--daemon-only`: both must
fail with "no running PM2 daemon", and no new `God Daemon ($PROBE)` process
may appear in `ps`. This shows that leftovers
of a daemon that died before the run never lead to `pm2` starting a new
one. The narrower race during a run is only detected, as described under
"Restart and post-deploy checks".

## Runtime environment

Production runs on the shared VM24 py311 pyenv, not on an environment built
from `uv.lock`. On 2026-10-24, 12 of 18 runtime packages differed from
`uv.lock`; for example dask/distributed 2025.3.0 vs 2025.11.0, numpy 2.2.4
vs 2.3.4, xarray 2025.3.1 vs 2025.10.1, fastapi 0.115.12 vs 0.121.1. pyTMD
2.2.8, scipy, numcodecs, and netCDF4 match. Repository tests run in the
`uv.lock` environment, so passing them locally does not verify py311;
release sequence step 2 runs them with py311.

On 2026-10-24, a PM2 restart whose PATH included `uv` started a `uv` sync
that created an incomplete `.venv` in the release directory; it was moved
(not deleted) to `/home/odbadmin/tmp/tide-uv-venv-incomplete-20261024`. The
hot patch made the launcher prefer py311, and the v0.3.2 target removes the
`uv run` and PATH fallbacks entirely.

## Operational Notes

- Launcher selection: see "Shared Dask TLS". The v0.3.2 target runs only an
  explicit absolute `GUNICORN_BIN` or the VM24 py311 Gunicorn; the hot patch
  currently on VM24 still has the `uv run` and PATH fallbacks.
- The launcher runs Gunicorn with `--reload`, so writing Python sources into
  the release directory reloads the live workers immediately.
- TPXO10 coastline reclassification can return no value where TPXO9 used to
  serve extrapolated values.
- Large bbox maps are capped by `MAX_BBOX_CELLS=500000` after applying
  `sample`.
- A non-blocking Polars CPU feature warning may appear in PM2 logs on VM24.

## Open Operational Items

- Follow-up slice (separate PR): remove Tide's Dask client. No Tide runtime
  path uses it (reads are direct); removing it would also allow retiring
  the `tide-client` identity.
- Make the py311 runtime reproducible (a lock or recorded freeze) or move
  production to an environment built from `uv.lock`.
- Parameterize host-specific paths in `conf/start_app.sh` (the launcher
  tests currently rewrite them in a sandbox copy).
- Remove `--reload` from the production launcher.
- Narrow the `tide` entry's `pre_stop` process match.
- Clean up merged branches and VM24 temporary directories once the above is
  settled.
