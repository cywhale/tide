# VM24 TPXO10 Deployment Notes

Initial TPXO10 cutover: 2026-06-18. Formal v0.3.1 release-directory
cleanup: 2026-07-08. Shared Dask mutual TLS on VM24: 2026-10-24 (host change
plus a hot patch of two Tide files). API version: v1.1.0.

- Released (tagged): v0.3.1.
- Target/pending release: v0.3.2 (PR #2, merge commit `ef780ea`). Not tagged
  and not deployed; see "v0.3.2 release sequence" and "Validation record".

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
- Public HTTPS frontend: port 443 for `eco.odb.ntu.edu.tw`, with a valid
  certificate. Verified-TLS checks of production go through 443.
- Internal Gunicorn listener: `127.0.0.1:8040`, HTTPS with the release
  directory's `conf/privkey.pem`/`fullchain.pem`. At the 2026-10-27
  validation it still served a certificate that expired on 2023-08-25. A
  verified-TLS request straight to 8040 therefore fails; do not point smoke
  checks at 8040, and never work around it with `curl -k`. Replacing that
  certificate is an open owner decision (see "Open Operational Items").
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
   the production PM2 daemon. Canary comparisons go through the public 443
   frontend (`--prod-port 443`), not the internal 8040 listener.
4. The owner gives separate authorization for the production update.
5. **Production deployment.** This changes the live service. Deploy only
   the v0.3.2 runtime manifest, one atomic rename per file (not one
   transaction across the files), then restart `tide`
   and run the post-deploy checks, as described in "Production deployment
   procedure" below.
6. Update this file with the outcome: rollback status "components
   validated" if the TPXO9 canary and the PM2 validation passed (see
   "Rollback"). Tag `v0.3.2` only after the validation and the production
   deployment are complete.

### Outstanding rollout checks

Each item stays open until it has been done once with its result recorded.
Results so far are in "Validation record" below.

- [ ] py311 test suite in a separate VM24 checkout (step 2). **Run; not
      green**: one failure (Zarr pin) plus known warnings. The owner
      authorized aligning Zarr only; the procedure is validated in
      isolation but not run on VM24 (see "Py311 Zarr alignment").
- [x] Canary certificate: SAN, expiry, and key match recorded. The curl
      version is implied by `https_smoke.sh` passing, since it refuses curl
      older than 7.76; the exact version was not recorded.
- [ ] `conf/replace_runtime_files.sh` rehearsed on a scratch copy on VM24
      (see "Production deployment procedure"). A local sandbox rehearsal
      passed after a fix (see the record); the VM24 run, with GNU
      coreutils and bash 5, is not yet reported.
- [x] Canary for the TPXO10 and TPXO9 stores: parts A, B, and C passed.
- [x] PM2 environment validation with an isolated `PM2_HOME`, including
      the `pm2_restart_check.sh` rehearsal on the dummy app.
- [ ] Production daemon liveness, read-only:
      `pm2_restart_check.sh --daemon-only --pm2-home /home/odbadmin/.pm2`
      reports the production daemon's PID. It never calls `pm2`, and it
      confirms that daemon's title is not truncated (see "PM2 environment
      validation").
- [ ] Canary symlink cleanup confirmed read-only. The cleanup command was
      sent, but its completion was not confirmed.
- [ ] Owner authorization for the production deployment (step 4).

### Validation record

Isolated validation of merge commit `ef780ea` on VM24 by Codex, reported
2026-10-27. Nothing was installed or upgraded, and production was not
changed.

- **py311 suite: not green.** One failure:
  `tests/test_app_smoke.py::test_pinned_runtime_versions` expects Zarr
  2.18.7 (the `uv.lock` pin), but production py311 has 2.18.6. Existing
  NumPy ABI (`numpy.ndarray size changed`) and Polars CPU-feature warnings
  also appear. The pin and py311 were deliberately left unchanged. The
  owner has since authorized aligning Zarr only, through a separate
  approved maintenance step; see "Py311 Zarr alignment" (not executed).
- **Canary certificate:** the Let's Encrypt certificate used for the canary
  (through a symlink in the canary checkout) is valid. Its SAN is
  `eco.odb.ntu.edu.tw`, it is valid until 2026-12-10, its private key
  matches, and `odbadmin` can read both.
- **HTTPS canary (part B), both passed with `conf/https_smoke.sh` from the
  merged commit**, with TLS verified throughout and no `curl -k`:
  - TPXO10: `--mode same --canary-port 8041 --prod-port 443`. HTTP 200, and
    the body is identical to the production 443 frontend.
  - TPXO9: `--mode different --canary-port 8041 --prod-port 443`. HTTP 200,
    and the body differs.
- **Production unaffected (part C):** the production Tide Gunicorn PIDs
  stayed 97326/97327/97328, release files were not modified, port 8041 was
  released, and no production PM2 operation was run.
- **Dask transport (part A): passed** during the TPXO10 canary run:
  - The scheduler's client list showed three `Client-tideapi-*` entries:
    the two production workers plus the one canary worker.
  - A separate one-off client with the same mTLS settings completed the
    trivial task and got `42`.
  - The canary log had no Dask connection-failure message.
- **PM2 environment validation:** isolated daemon, short `PM2_HOME`, and a
  `.cjs` dummy config (see "PM2 environment validation"). All of these
  passed:
  - The TPXO9 and TPXO10 `TIDE_ZARR_PATH` values were passed through
    correctly.
  - After the variable was unset, `restart --update-env` kept the old value.
    This confirms that rollback must always set the store explicitly.
  - `save`, `kill`, and `resurrect` preserved the value.
  - `pm2_restart_check.sh` succeeded, refused a wrong executable, passed
    `--daemon-only`, and failed closed after the daemon was stopped.
- **Rollback:** components validated (2026-10-27); production rollback not
  exercised. See "Rollback".
- **`replace_runtime_files.sh`, local sandbox rehearsal (2026-10-27, not on
  VM24):** the real script was run against a temporary git checkout and a
  temporary release directory with manifest modes 750 and 640. It used
  macOS bash 3.2 and BSD `stat`, so the VM24 run is still required.
  - It found a defect. Each staged copy was checked only when it was staged.
    A copy altered after that check, while the next file was being staged
    or after the first rename, was still renamed into the release directory.
    Only the post-rename check caught it, with the altered file already
    live. The script now checks every staged copy again before the first
    rename, and each one again right before its own rename. Regression
    tests are in `tests/test_ops_scripts.py`.
  - With the fix, every scenario behaved as expected:
    - `deploy`: exit 0. Both files match the commit blobs, the modes are
      kept, other runtime files are unchanged, and `staging/` is removed.
    - An existing `--work`: exit 1, "already exists". No release file
      changed, and the earlier backup was kept.
    - A staged copy altered during staging: exit 1, "nothing was replaced".
      No file was renamed.
    - A checkout file that differs from the commit: exit 1, "nothing was
      replaced".
    - A staged copy altered after the first rename: exit 1, naming
      `restore`. The launcher was replaced and the client was not. `restore`
      then exited 0 with the original contents and modes.
    - `restore` after a successful `deploy`: exit 0, with the original
      contents and modes.
  - The temporary directory was removed afterwards.
- **Canary symlinks:** a cleanup command was sent, but its completion was
  not confirmed. Treat them as possibly present until checked read-only.

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
- stages every manifest file and verifies it against its blob at the commit,
  then verifies every staged copy again before replacing any, so a mismatch,
  including a staged copy altered after its first check, replaces nothing;
- swaps each file in with its own rename, verifying the staged copy right
  before the rename and the target after it;
- confirms that the other runtime files are unchanged.

`tests/test_ops_scripts.py` exercises these refusals, the two-phase
behavior, the rename, a staged copy altered after its first check (before
the first rename, and between the renames), recovery by `restore` after a
failure between the two renames, and `restore`'s record checks locally.

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

The production smoke goes through the public HTTPS frontend
(`--prod-port 443`). The internal 8040 listener still serves an expired
certificate (see "Current Production Layout"), so a verified request to
8040 would fail; do not use `--prod-port 8040`, which is the script's
default.

```bash
HOST=eco.odb.ntu.edu.tw   # the SAN hostname recorded before the canary
"$SRC/conf/pm2_restart_check.sh" --pm2-home /home/odbadmin/.pm2 --name tide \
  && "$SRC/conf/https_smoke.sh" --host "$HOST" --mode prod --prod-port 443 \
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

Status: **rollback components validated (2026-10-27); production rollback
not exercised.**

The 2026-10-27 validation met the condition below: the TPXO9 canary passed
(the data path through the current runtime), and so did the isolated PM2
environment validation (`TIDE_ZARR_PATH` propagation, including
save/kill/resurrect, on a dummy app). The production `tide` entry was never
switched. This status does not mean that a production rollout or a
production rollback drill has been done.

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
- Use a currently valid certificate and key for `eco.odb.ntu.edu.tw` (the
  Let's Encrypt certificate used in the 2026-10-27 validation), referenced
  by absolute path or a symlink in the canary checkout; never copy the key.
  Do not use the release directory's `conf/privkey.pem`/`fullchain.pem`:
  that certificate expired on 2023-08-25, so verified smoke requests to the
  canary would fail.

Start and stop:

- Set `SHA` to the merged commit under test, and `LE_PRIVKEY` and
  `LE_FULLCHAIN` to the valid Let's Encrypt key and certificate (the same
  paths checked in part B, step 1). Then run the block below. It runs under
  `set -euo pipefail` in a subshell, so any failure stops it before
  Gunicorn starts, while the operator's own shell keeps its options and
  gets none of the exported variables (no stray `TIDE_ZARR_PATH`). It:
  - requires `SHA` (7 to 40 hex digits), `LE_PRIVKEY`, and `LE_FULLCHAIN`;
  - changes into the canary checkout, and refuses unless that checkout's
    `git rev-parse HEAD` starts with `SHA` and it has no tracked changes;
  - exports the same Dask TLS variables as `conf/start_app.sh` plus the
    absolute `TIDE_ZARR_PATH`, and refuses unless every file is readable;
  - starts Gunicorn without `--reload`, with `--chdir` pinned to the
    checkout, and prints its PID.

  ```bash
  CANARY_PID=$(
    set -euo pipefail
    : "${SHA:?set SHA to the merged commit under test}"
    : "${LE_PRIVKEY:?set LE_PRIVKEY to the LE private key path}"
    : "${LE_FULLCHAIN:?set LE_FULLCHAIN to the LE fullchain path}"
    [[ $SHA =~ ^[0-9a-f]{7,40}$ ]] \
      || { echo "SHA must be 7-40 hex digits: $SHA" >&2; exit 1; }
    CHECKOUT=/home/odbadmin/tmp/tide-v0.3.2-$SHA   # the canary checkout
    cd "$CHECKOUT"
    head=$(git rev-parse HEAD)
    [[ $head == "$SHA"* ]] \
      || { echo "checkout is at $head, not $SHA" >&2; exit 1; }
    git diff --quiet HEAD -- \
      || { echo "checkout has tracked changes" >&2; exit 1; }
    DASK_TLS_DIR=/home/odbadmin/.config/dask/tls   # as in conf/start_app.sh
    export DASK_SCHEDULER_ADDRESS=tls://127.0.0.1:8786
    export DASK_TLS_CA_FILE="$DASK_TLS_DIR/certs/ca.crt"
    export DASK_TLS_CLIENT_CERT="$DASK_TLS_DIR/certs/tide-client.crt"
    export DASK_TLS_CLIENT_KEY="$DASK_TLS_DIR/private/tide-client.key"
    # Store under test: TPXO10 here; for the rollback canary use
    # /home/odbadmin/python/tide/data/tpxo9.zarr.
    export TIDE_ZARR_PATH=/home/odbadmin/python/tide_tpxo10_atlas_v2/data/tpxo10.zarr
    for f in "$DASK_TLS_CA_FILE" "$DASK_TLS_CLIENT_CERT" \
             "$DASK_TLS_CLIENT_KEY" "$TIDE_ZARR_PATH" \
             "$LE_PRIVKEY" "$LE_FULLCHAIN"; do
      [ -r "$f" ] || { echo "not readable: $f" >&2; exit 1; }
    done
    /home/odbadmin/.pyenv/versions/py311/bin/gunicorn tide_app:app \
      --chdir "$CHECKOUT" -w 1 \
      --worker-class uvicorn.workers.UvicornWorker -b 127.0.0.1:8041 \
      --timeout 180 --keyfile "$LE_PRIVKEY" --certfile "$LE_FULLCHAIN" \
      < /dev/null > "$CHECKOUT-canary.log" 2>&1 &
    echo "$!"
  ) && echo "canary PID $CANARY_PID"
  ```

  If it prints an error instead of a PID, nothing was started; fix the
  cause and rerun it. The canary log is
  `/home/odbadmin/tmp/tide-v0.3.2-$SHA-canary.log`. Record the PID, and
  stop the canary with `kill "$CANARY_PID"` only.
- After the canary, remove the symlinks created for it (certificate links,
  and store links from step 2 if they are no longer needed). Confirm the
  removal read-only, for example with `ls -l` on the canary checkout, and
  record it. A cleanup command without a confirmed result does not count.

Acceptance has three separate parts.

A. Dask transport (TLS). This shows the canary's TLS settings and the
`tide-client` identity work with the scheduler and worker. It does not test
Tide computation, because the Tide API never submits Dask tasks.

1. Before starting, use a one-off client with the same four Dask TLS
   variables as the canary block (it exports them only inside its
   subshell, so set them for the client separately) to record the
   scheduler's client list:
   `client.run_on_scheduler(lambda dask_scheduler: sorted(dask_scheduler.clients))`.
   Production holds one `Client-tideapi-*` entry per Gunicorn worker.
2. While the canary runs, the list gains exactly one `Client-tideapi-*`
   entry, the scheduler log shows `Receive client connection:
   Client-tideapi-*` at the canary start time, and the canary log has no
   `Dask client connection failed` warning.
3. The same one-off client completes a trivial worker task, for example
   `client.submit(lambda x: x + 1, 41).result()` returning `42`.
4. After `kill "$CANARY_PID"`, the client list returns to the step 1
   baseline.

B. Tide API and store. This shows the canary serves the intended store.
Verify HTTPS against the certificate's hostname instead of disabling
verification; never use `curl -k`.

1. Check the canary certificate: its hostname, its expiry, and that the key
   matches it. Use the same `LE_FULLCHAIN` and `LE_PRIVKEY` as the canary
   start command; run this check before starting the canary.

   ```bash
   openssl x509 -in "$LE_FULLCHAIN" -noout -ext subjectAltName -enddate
   openssl x509 -in "$LE_FULLCHAIN" -noout -pubkey | openssl sha256
   openssl pkey -in "$LE_PRIVKEY" -pubout | openssl sha256   # must match
   ```

   The commands below assume `eco.odb.ntu.edu.tw`; if the certificate lists
   a different name, use that name.
2. Run `conf/https_smoke.sh` from the canary checkout. It queries the canary
   (8041) and production through the public HTTPS frontend
   (`--prod-port 443`, a read-only GET). Always pass `--prod-port 443`
   explicitly: the script defaults to 8040, whose certificate has expired.
   Both requests go through `curl --resolve HOST:PORT:127.0.0.1` with
   verification on and
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
   # TPXO10 canary: body must be identical to production (443 frontend).
   "$SMOKE" --host eco.odb.ntu.edu.tw --mode same \
     --canary-port 8041 --prod-port 443 --out-root "$ROOT"
   # TPXO9 canary: both 200, bodies must differ.
   "$SMOKE" --host eco.odb.ntu.edu.tw --mode different \
     --canary-port 8041 --prod-port 443 --out-root "$ROOT"
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
- Name the dummy config `envprobe.config.cjs`, not `.js`.
  `/home/odbadmin/tmp/package.json` puts `.js` files under
  `/home/odbadmin/tmp` in ES module scope, and PM2 loads ecosystem files with
  `require()`. A `.js` config there fails with `ERR_REQUIRE_ESM`, which the
  2026-10-27 validation hit.
- Keep `PM2_HOME` short. The liveness check matches the daemon title
  `God Daemon (<PM2_HOME>)`, and with a long path that title appears
  truncated in `ps`, so the check refuses the daemon. That refusal is safe;
  fix it by using a shorter path, never by loosening the check.

```bash
PROBE=/home/odbadmin/tmp/pm2p   # keep short: see above
PM2=$HOME/.npm-global/bin/pm2
mkdir -p "$PROBE"
cat > "$PROBE/envprobe.config.cjs" <<'JS'
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
  "$PM2" start "$PROBE/envprobe.config.cjs"; probe
PM2_HOME="$PROBE" "$PM2" delete envprobe
PM2_HOME="$PROBE" TIDE_ZARR_PATH=/home/odbadmin/python/tide_tpxo10_atlas_v2/data/tpxo10.zarr \
  "$PM2" start "$PROBE/envprobe.config.cjs"; probe
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
version titles its daemon `God Daemon (<PM2_HOME>)`; the 2026-10-27
validation confirmed this with a short `PM2_HOME`. Record the output.

Before step 5, run the same read-only check against the production daemon:
`pm2_restart_check.sh --daemon-only --pm2-home /home/odbadmin/.pm2`. It
never calls `pm2`, and it must report the production daemon's PID. That
confirms the production title is not truncated, so the real restart check
will accept it.

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

The 2026-10-27 py311 run (step 2) was **not green**: the Zarr version pin
test failed (py311 has 2.18.6, `uv.lock` pins 2.18.7), with the known NumPy
ABI and Polars CPU-feature warnings. Do not hide this by changing the pin
or installing packages into the shared py311. The owner has authorized
aligning only Zarr to 2.18.7; the validated procedure and what must still
be confirmed first are in "Py311 Zarr alignment" below. It has not been
run on VM24.

On 2026-10-24, a PM2 restart whose PATH included `uv` started a `uv` sync
that created an incomplete `.venv` in the release directory; it was moved
(not deleted) to `/home/odbadmin/tmp/tide-uv-venv-incomplete-20261024`. The
hot patch made the launcher prefer py311, and the v0.3.2 target removes the
`uv run` and PATH fallbacks entirely.

## Py311 Zarr alignment

Status: **proposed, not executed.** The owner authorized aligning the
shared py311 Zarr from 2.18.6 to the `uv.lock` pin 2.18.7, changing no
other package. This is a production change to an interpreter shared by
several services. The items under "Preconditions before any production
run" are prerequisites, not formalities: none is settled yet. Do not run
`apply` or `rollback` on VM24 until every one is settled and recorded here,
in an approved maintenance window.

### Isolated validation (2026-10-27, local, not on VM24)

- py311 was read on VM24 without changes: package metadata through
  `python -I -B` (pip was not run there), `ps`, `/proc`, and
  `crontab -l`. It is Python 3.11.4 on x86_64 (glibc 2.39, Ubuntu 24.04)
  with pip 25.3 and 240 distributions. Some are not Tide's (for example a
  local `odbargo_app` tarball and an editable `woa23`). There are also four
  orphaned dist-info directories that hold only `REQUESTED`, and duplicate
  dist-info entries for netCDF4 and h5py.
- A linux/amd64 `python:3.11.4` container with pip 25.3 reproduced 231 of
  the 234 named packages at identical versions, installed with `--no-deps`.
  The three it could not reproduce are `odbargo_app`, the editable `src`,
  and `connectorx` 0.2.3, which has no Python 3.11 build on PyPI. It
  reproduced the production failure exactly: the pin test failed, with the
  same NumPy ABI warning.
- Production's zarr 2.18.6 files are byte-identical to the PyPI wheel (44
  file hashes). Between the 2.18.6 and 2.18.7 wheels, the only changed file
  is `zarr/version.py`, which holds the version string. The only metadata
  change is a new `numcodecs<0.16` bound, which 0.15.1 satisfies. The 2.18.7
  wheel matches the `uv.lock` hash.
- `pip install zarr==2.18.7` with the resolver enabled changed only the
  zarr line in the freeze. `pip check` showed the same two existing,
  unrelated conflicts (argopy needs older xarray; dask-expr needs another
  dask). Test suite results:
  - before, on 2.18.6: 1 failed (the pin test), 192 passed;
  - after, on 2.18.7: 193 passed;
  - after rolling back to 2.18.6: the same result as before, and the freeze
    identical to the original.
- An MHW-style round trip passed in both directions: a group written by
  xarray on 2.18.6, appended with `append_dim` on 2.18.7, and read back on
  2.18.6.
- `conf/py311_zarr_align.sh` wraps the procedure. It is fail-closed and
  covered by `tests/test_ops_scripts.py`, including a pip failure whose
  output a `tee` still captures, failed and zarr-less snapshots, drift since
  `prepare`, and changes beyond zarr. It also covers `pip check` results
  that are not in pip's format, such as a traceback with exit 1, unknown,
  mixed, or empty output, or a stderr warning, including the case where
  `prepare` and `apply` fail the same way. `tests/test_deployment_doc.py`
  runs the runbook's own `apply` and `rollback` chains and confirms that
  the restart and smoke never run after any of these failures.

### Preconditions before any production run

Each item below is open. The read-only checks of 2026-10-27 and
2026-10-28 read only the crontab, scripts, configuration, logs, and `ps`.
They did not source or run `update_mhw.sh`, ran no `pm2` command, and
changed nothing. Their evidence is noted with each item, together with
what it does not prove.

- [ ] **MHW API interpreter after a restart.**
      - Observed: PM2 app `mhwapi` (from `~/.pm2/dump.pm2`, saved
        2026-10-24) runs `/home/odbadmin/proj/marineheatwave/API/conf/start_app.sh`
        with `bash`, in fork mode, with `autorestart`, from working
        directory `.../marineheatwave/API`.
      - Observed: that launcher exports the `mhw-client` Dask TLS identity,
        then starts a bare `gunicorn mhw_app:app ... -b 127.0.0.1:8030
        --reload`. It does not use `exec`, so PM2's PID is the `bash`
        launcher and Gunicorn is its child.
      - Observed: the running Gunicorn (started 2026-10-24) is
        `/home/odbadmin/.pyenv/versions/py311/bin/gunicorn`.
      - Not proven: which `gunicorn` the launcher finds after a restart.
        That depends on the `PATH` PM2 has stored for `mhwapi`. This is the
        same `PATH` dependence that Tide v0.3.2 removed from its own
        launcher.
- [ ] **`update_mhw.sh` cron interpreter.**
      - Observed: crontab entry `50 4 18 * *`, at 04:50 server time on the
        18th of each month. The crontab sets no `PATH` or `SHELL`. The next
        run is due 2026-11-18 04:50.
      - Observed: the script initializes pyenv, runs `pyenv activate
        py311`, and then runs `update_mhw.py` with
        `/home/odbadmin/.pyenv/shims/python`.
      - Observed: the pyenv global version is `py311`, pyenv-virtualenv is
        installed, and there is no `.python-version` in the API directory
        or any parent.
      - Inferred, not observed: every way pyenv can resolve that shim leads
        to py311. The log records only the shim path, never the
        interpreter it resolved to.
      - Observed: the last run (2026-10-18 04:50–04:51) appended to
        `data/mhw.zarr` with xarray and Dask arrays. It uses no distributed
        Dask client.
      - Observed: the script ends by running `pm2 restart mhwapi` and
        restarting nginx, so this cron is also a monthly MHW restart. The
        window must not overlap it.
- [ ] **MHW restart procedure.**
      - No maintainer procedure was found. The MHW `README.md` and
        `change_log.md` describe none; the change log only mentions a PM2
        `pre_stop`.
      - The only restart in use is the cron's own `pm2 restart mhwapi`.
      - Tide's `pm2_restart_check.sh` is not validated for `mhwapi`. Its
        `--expect-exec` check reads the command line of PM2's PID, which
        for MHW is the `bash` launcher, not Gunicorn.
      - The repository's `conf/ecosystem.config.js` does not match the
        running setup: it starts `dask-worker` on plain `tcp://`, and it
        has a `pre_stop` that runs `kill -9` on `mhw_app`.
      - Unconfirmed. The MHW maintainer must supply the procedure.
- [ ] **MHW health check and smoke.**
      - There is no health endpoint. The read-only GET routes are
        `/api/mhw` (JSON), `/api/mhw/csv`, `/api/mhw/plot`, and
        `/api/swagger/mhw`.
      - nginx proxies the public 443 paths `/api/mhw` and `/api/swagger/mhw`
        on `eco.odb.ntu.edu.tw` to `https://127.0.0.1:8030`.
      - The certificate served on 8030 (`API/conf/fullchain.pem`) expired
        on 2023-08-25, so a verified smoke has to go through 443, as Tide's
        does.
      - Unconfirmed: no query or expected result is defined. The MHW
        maintainer must supply them.
- [ ] **Maintenance window.** It is not set. It must avoid the MHW cron
      (04:50 on the 18th, with its own MHW restart). The owner and the MHW
      maintainer agree on it.
- [ ] **Dask scheduler and worker.** They run on py311 without `--reload`
      and keep the old Zarr code in memory until they restart. The code is
      identical, so a later restart under the host runbook is enough.
      Confirm with the owner.
- [ ] **How Tide and MHW are handled during the install: undecided.**
      Both run with `--reload`, so `apply` and `rollback` reload their
      workers as soon as the zarr files change, not at the next restart. A
      worker that reloads while pip is swapping the files could fail to
      boot. Neither option below is decided or validated on VM24:
      - install while the services run, then restart each one straight
        away with its own procedure;
      - stop Tide and MHW, install, then start them. Validated tooling does
        not cover stopping and starting them.

      The chains below show only the order of the script, Tide's restart
      check, and Tide's smoke. They do not choose between these options,
      and they do not include MHW. Revise them once the owner decides.
- [ ] **Production `pip check` output.** It has never been run on VM24.
      The script accepts only pip check's two result formats (see
      "Procedure"). If `prepare` rejects production's output, for example
      a metadata warning caused by the orphaned dist-info directories, stop
      and report it with `pip-check-before.txt.rejected` and
      `pip-check-before.txt.stderr`. Never loosen the check.

### Procedure

Use `conf/py311_zarr_align.sh` from a merged checkout that contains it
(`SRC`). It runs under `set -euo pipefail`, independent of the operator's
shell. Every step either succeeds or exits non-zero before the next one. A
failed `pip install` is not hidden by the `tee` that logs it, and a failed
snapshot leaves no freeze file behind. Nothing chained after it with `&&`
runs on failure.

`pip check` is accepted only in one of the two forms pip 25.3 produces:

- exit 0 with stdout exactly the one line `No broken requirements found.`
  and its newline, compared byte for byte, so that no extra blank line,
  missing newline, `\r`, or trailing space gets through;
- exit 1 with one or more newline-terminated conflict lines, no blank line,
  and nothing else. Each line must be one of `X v requires Y, which is not installed.`,
  `X v has requirement R, but you have Y v.`, or
  `X v is not supported on this platform`.

Both forms also need empty stderr. Anything else is treated as pip check
failing, and the script stops without recording a result. That includes a
traceback, empty or unknown output, a warning (pip also exits 1 on
unreadable metadata), and any other exit status. Only an accepted result
goes into the before/after comparison.

1. Before the window, `prepare` changes nothing in py311. It records the
   sorted `pip freeze --all`, the accepted `pip check` result with its exit
   status, and the Zarr version. It downloads the 2.18.7 and 2.18.6 wheels pinned by SHA-256
   into `$D/wheels`, and dry-runs the install, which must offer exactly
   `zarr-2.18.7`.

   ```bash
   SRC=/home/odbadmin/tmp/tide-v0.3.2-$SHA   # merged checkout with the script
   ALIGN="$SRC/conf/py311_zarr_align.sh"
   D=/home/odbadmin/tmp/zarr-align-$(date +%Y%m%d)   # must not exist yet
   "$ALIGN" prepare --dir "$D"
   ```

2. In the window, and only after every precondition above is settled,
   `apply` refuses to start unless py311 still matches the `prepare`
   freeze. It installs 2.18.7 offline with `--no-deps`, `--no-index`, and
   `--require-hashes`. Then it requires:
   - the freeze to differ only in `zarr==2.18.6` → `zarr==2.18.7`;
   - an accepted `pip check` result identical to the `prepare` one;
   - zarr to import as 2.18.7.

   Only then do Tide's restart check and smoke run. The service-handling
   decision above may change this chain:

   ```bash
   "$ALIGN" apply --dir "$D" \
     && "$SRC/conf/pm2_restart_check.sh" --pm2-home /home/odbadmin/.pm2 --name tide \
     && "$SRC/conf/https_smoke.sh" --host eco.odb.ntu.edu.tw --mode prod \
          --prod-port 443 --out-root "$D-smoke"
   ```

3. After `PASS prod`, restart and smoke-test MHW with its own procedure.
   That procedure does not exist in this runbook yet; it is a
   precondition. Then run the py311 test suite from the merged checkout
   (release sequence step 2). The pin test should now pass.

If `apply` fails, do not restart or smoke anything. Report to the owner
with `$D/install.log` and the snapshots. The state of py311 after a failed
install is unknown, and rolling back is the owner's decision.

What to check and record:

- the versions: zarr 2.18.6 → 2.18.7, with numcodecs 0.15.1 and numpy
  2.2.4 unchanged;
- that the freeze differs only in the zarr line (`freeze-before.txt` vs
  `freeze-after.txt`);
- that `pip check` was accepted both times and is unchanged
  (`pip-check-before.txt` vs `pip-check-after.txt`);
- Tide's `OK restart` and `PASS prod`, and MHW's smoke result.

Rollback to 2.18.6 uses the same `$D`. It installs the hash-pinned 2.18.6
wheel saved by `prepare`, offline. It requires the freeze and an accepted
`pip check` result to match the `prepare` snapshot exactly. Restart and
smoke-test the services the same way, MHW included, under the same
preconditions:

```bash
"$ALIGN" rollback --dir "$D" \
  && "$SRC/conf/pm2_restart_check.sh" --pm2-home /home/odbadmin/.pm2 --name tide \
  && "$SRC/conf/https_smoke.sh" --host eco.odb.ntu.edu.tw --mode prod \
       --prod-port 443 --out-root "$D-rollback-smoke"
```

A directory accepts one `apply`. To try again after a rollback, run
`prepare` with a new directory.

## Operational Notes

- Launcher selection: see "Shared Dask TLS". The v0.3.2 target runs only an
  explicit absolute `GUNICORN_BIN` or the VM24 py311 Gunicorn; the hot patch
  currently on VM24 still has the `uv run` and PATH fallbacks.
- The launcher runs Gunicorn with `--reload`, so writing Python sources into
  the release directory reloads the live workers immediately.
- HTTPS: the public frontend on 443 has a valid certificate; the internal
  Gunicorn listener on 8040 still serves the release directory's
  certificate, which expired on 2023-08-25. Point verified-TLS checks at
  443. A `tide` restart reloads the same expired certificate on 8040; it
  does not fix it.
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
  production to an environment built from `uv.lock`. Until then the py311
  suite fails `test_pinned_runtime_versions` (Zarr 2.18.6 vs 2.18.7). The
  owner authorized aligning Zarr only; see "Py311 Zarr alignment" for the
  procedure and the items still to confirm.
- The internal 8040 listener's certificate expired on 2023-08-25 (the public
  443 frontend is valid). Whether and how to replace it is an owner
  decision; it is not part of the v0.3.2 runtime manifest.
- `conf/https_smoke.sh` defaults to `--prod-port 8040`; consider changing
  the default to 443 so the safe choice is the default. For now, the
  runbook passes `--prod-port 443` explicitly.
- Parameterize host-specific paths in `conf/start_app.sh` (the launcher
  tests currently rewrite them in a sandbox copy).
- Remove `--reload` from the production launcher.
- Narrow the `tide` entry's `pre_stop` process match.
- Clean up merged branches and VM24 temporary directories once the above is
  settled.
