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
      authorized aligning Zarr only. Its `prepare` ran on VM24 on
      2026-10-28; `apply` waits for the open items of the run sheet (see
      "Py311 Zarr alignment").
- [x] Canary certificate: SAN, expiry, and key match recorded. The curl
      version is implied by `https_smoke.sh` passing, since it refuses curl
      older than 7.76; the exact version was not recorded.
- [x] `conf/replace_runtime_files.sh` rehearsed on a scratch copy on VM24
      (2026-10-28, script from PR #3 at `ca304d9`, bash 5.2 with GNU
      coreutils): `deploy`, refusal of a reused `--work`, refusal of a
      staged copy altered after its check, and `restore` all passed. The
      production files were unchanged; see the record.
- [x] Canary for the TPXO10 and TPXO9 stores: parts A, B, and C passed.
- [x] PM2 environment validation with an isolated `PM2_HOME`, including
      the `pm2_restart_check.sh` rehearsal on the dummy app.
- [x] Production daemon liveness, read-only (2026-10-28):
      `pm2_restart_check.sh --daemon-only --pm2-home /home/odbadmin/.pm2`
      printed `daemon alive: PID 3352` and `OK daemon-only: pm2 was not
      called`, and exited 0. `ps` shows the full, untruncated title
      `PM2 v5.4.2: God Daemon (/home/odbadmin/.pm2)`.
- [x] Canary symlink cleanup (2026-10-28). A read-only check earlier that
      day found the earlier cleanup had not happened. Afterwards, the four
      links in `/home/odbadmin/tmp/tide-v032-codex-20261027` were each
      removed with `unlink`, which does not follow a link:
      - `canary-validation/tls/privkey.pem` and
        `canary-validation/tls/fullchain.pem`, which pointed into
        `/etc/letsencrypt/live/eco.odb.ntu.edu.tw/`;
      - the step 2 store links `data/tpxo9.zarr` and `data/tpxo10.zarr`.

      A read-only check confirmed that none of the four remains and that no
      symlink is left under that checkout. The type, inode, and mtime of
      every link target were the same before and after, and the release's
      runtime hashes were unchanged.
- [ ] Owner authorization for the production deployment (step 4). The
      2026-10-28 authorization covers only the py311 Zarr change, not
      these runtime files.

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
- **Canary symlinks: removed on 2026-10-28.** A read-only check found the
  two certificate links and the two store links still present. They were
  then removed with `unlink`, and the result was confirmed read-only (see
  "Outstanding rollout checks"). No certificate content was read. No link
  target was changed.

Checks on VM24 on 2026-10-28, by Claude at the owner's request, using the
PR #3 scripts at `ca304d9`. They were copied to
`/home/odbadmin/tmp/tide-pr3-ca304d9-ops/conf/`, and their sha256 hashes
matched the commit. No service was stopped, started, or restarted, and no
`pm2` command was run.

- **`replace_runtime_files.sh` scratch rehearsal: passed.**
  - `REL` was a new `mktemp` directory under `/home/odbadmin/tmp`, on the
    same filesystem as the release (device 64512). The rehearsal script
    refused any `REL` that resolved to the release directory.
  - `REL` held copies of the release's 15 runtime files, made with
    `cp -p`; no `*.pem` was copied. `SRC` was the merged checkout
    `/home/odbadmin/tmp/tide-v032-codex-20261027` at `ef780ea`, with no
    tracked changes; it was only read.
  - `deploy`: exit 0. Both manifest files matched the commit blobs, and
    modes 775 and 664 were kept. The other runtime files were unchanged, and
    `staging/` was removed.
  - The same `--work` again: exit 1, "already exists". Hashes, modes, and
    inodes in `REL` were unchanged.
  - `restore`: exit 0. `REL` was back to its original contents and modes.
  - A staged copy altered after its check (through a `chmod` wrapper on
    `PATH`): exit 1, "changed after verification; nothing was replaced".
    Hashes, modes, and inodes in `REL` were unchanged.
  - The release's own manifest hashes were the same before and after.
  - The variant that deploys from an edited source clone was not run on
    VM24; local tests cover it.
  - Cleanup: all 35 scratch files were removed one by one. The empty
    directories were removed afterwards, outside this session. A read-only
    check later on 2026-10-28 found no
    `/home/odbadmin/tmp/tide-replace-rehearsal.1Xhks1` and no other
    `tide-replace-rehearsal.*` entry in `/home/odbadmin/tmp`.
- **Release runtime hashes around the cleanup:** the 15 runtime files of
  `/home/odbadmin/python/tide_tpxo10_atlas_v2` were hashed before and after
  the symlink cleanup, and the two lists were identical (list digest
  `a1de99c748a2ee1c…`). The manifest files are still the v0.3.1 hot-patch
  versions (`conf/start_app.sh` `8ee7b44d…`,
  `src/dask_client_manager.py` `d74c6690…`). The Tide Gunicorn PIDs are
  still 97326/97327/97328.
- **Production PM2 daemon, read-only:** alive as PID 3352 with an
  untruncated title (see "Outstanding rollout checks").
- **Zarr alignment `prepare`:** passed. See "Py311 Zarr alignment".
- **MHW interpreters and smoke baseline (later on 2026-10-28, read-only):**
  - the MHW API after a restart and the `update_mhw.sh` cron both resolve
    to py311;
  - the cron's own `pm2 restart mhwapi` has failed every month;
  - the proposed MHW check passed against the running service.

  See "MHW facts for the maintenance" and "MHW restart and post-restart
  smoke".

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

# Roll back to TPXO9. `pm2 delete tide` signals only the tide process tree;
# PM2 5.4.2 never runs the entry's pre_stop (see "PM2 and Dask lifecycle facts").
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
- Never start it through PM2 or the `tide` entry, which is the production
  app. The entry's `pre_stop` would kill every process whose command line
  contains `tide_app`, the canary and production alike; PM2 5.4.2 never runs
  it (see "PM2 and Dask lifecycle facts"), but do not rely on that. Do not
  use `conf/start_app.sh` either: it binds `127.0.0.1:8040` and enables
  `--reload`.
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
- The dummy app is not a `tide_app` process and has no `pre_stop`. Never
  use the production `conf/ecosystem.config.js` for this test: it defines
  the production `tide` app, and a separate `PM2_HOME` would not sandbox its
  `pre_stop` under a PM2 version that ran it (5.4.2 does not).
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
or installing packages into the shared py311 outside the approved
maintenance. The owner has authorized aligning only Zarr to 2.18.7. Its
`prepare` has run on VM24, and its `apply` has not; the run sheet and what
must still be settled first are in "Py311 Zarr alignment" below.

On 2026-10-24, a PM2 restart whose PATH included `uv` started a `uv` sync
that created an incomplete `.venv` in the release directory; it was moved
(not deleted) to `/home/odbadmin/tmp/tide-uv-venv-incomplete-20261024`. The
hot patch made the launcher prefer py311, and the v0.3.2 target removes the
`uv run` and PATH fallbacks entirely.

## Py311 Zarr alignment

Status: **`prepare` done on 2026-10-28; `apply` not run.** On 2026-10-28
the owner authorized the production maintenance for one package change:
the shared py311 Zarr from 2.18.6 to the `uv.lock` pin 2.18.7. No other
package may change. This authorization does not cover deploying the v0.3.2
runtime files (release sequence step 4). It is a production change to an
interpreter shared by Tide, MHW, and the Dask scheduler and worker. Run it
only from "Production run sheet" below, in an agreed window, after every
open item in its checklist is settled and recorded here.

`prepare` on VM24 (2026-10-28 10:34 +0800) used
`/home/odbadmin/tmp/tide-pr3-ca304d9-ops/conf/py311_zarr_align.sh` from
`ca304d9`. It exited 0 and py311 was unchanged: zarr still imports as
2.18.6. Its artifacts are in `/home/odbadmin/tmp/zarr-align-20261028T103416`:

- `freeze-before.txt`: 234 lines, including `zarr==2.18.6`. Its stderr was
  empty; the orphaned dist-info directories caused no warning.
- `pip-check-before.txt`: exit 1 with four existing conflicts, all in pip
  check's format and accepted. `odbargo-app` and `connectorx` require
  `numpy<2`, `dask-expr` requires `dask==2024.12.0`, and `argopy` requires
  `xarray<2024.3`. None involves zarr. The first two did not appear in the
  isolated run, which could not reproduce those two packages.
- `dry-run.txt`: `Would install zarr-2.18.7` and nothing else.
- `wheels/`: zarr 2.18.7 (sha256 `ac3dc403…`) and 2.18.6 (sha256
  `a5334311…`), both matching the hashes pinned in the script.
- `prepared`: the marker that `apply` requires. `apply` takes a new freeze
  and refuses to install if py311 differs from this snapshot.

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

### MHW facts for the maintenance (read-only, 2026-10-28)

These checks read `ps`, `/proc`, `ss`, PM2's pid files, dump, and log, PM2's
installed source, the MHW launcher and app source, the interpreter lines of
`update_mhw.sh`, its log, and its cron mail. Of the environment, they read
only path variables. pyenv's resolution was run read-only: `pyenv which`,
plus `pyenv sh-activate`, whose output was printed and never evaluated. No
`pm2` command was run, and nothing was started, stopped, or changed.

- **MHW API interpreter after a restart: py311.** Resolved read-only:
  - PM2's PID for `mhwapi` is the launcher
    `bash /home/odbadmin/proj/marineheatwave/API/conf/start_app.sh` (PID
    96946, started 2026-10-24 13:45:56). It runs a bare `gunicorn` without
    `exec`, so Gunicorn (PID 96947) is its child.
  - A `pm2 restart` without `--update-env` starts the launcher with the
    environment PM2 stored at its last start, which is the launcher's
    current environment.
  - In that `PATH`, the first `gunicorn` is `/home/odbadmin/.pyenv/shims/gunicorn`.
    The entries before it contain no `gunicorn`, and two of them no longer
    exist.
  - There is no `PYENV_VERSION`, and no `.python-version` from the API
    directory upward. The pyenv global version is `py311`. pyenv therefore
    resolves the shim to `/home/odbadmin/.pyenv/versions/py311/bin/gunicorn`,
    whose shebang is `py311/bin/python3.11` (`3.11.4/envs/py311`).
  - Replaying the launcher's path variables with `env -i` from the API
    directory gave the same result, and so does the running Gunicorn.
  - This holds only for a restart without `--update-env`, which
    `pm2_restart_check.sh` never passes, and only while the launcher is
    still PID 96946. A restart with `--update-env` would take the caller's
    `PATH`.
  - `pm2 resurrect` would use `~/.pm2/dump.pm2` (saved 2026-10-24 14:41),
    whose `PATH` also puts the pyenv shims first.
  - The post-restart check (`mhw_proc_check`) verifies the interpreter
    again.
- **`update_mhw.sh` cron interpreter: py311.** Resolved read-only:
  - Cron runs the script directly (`#!/bin/bash`). The script `cd`s to the
    API directory, puts `~/.pyenv/bin` on `PATH`, and evaluates
    `pyenv init -` and `pyenv virtualenv-init -`. It then runs
    `pyenv activate py311` and starts `update_mhw.py` through
    `/home/odbadmin/.pyenv/shims/python`.
  - `pyenv activate py311` exports `PYENV_VERSION=py311`. In a cron-like
    `env -i` from the API directory, `pyenv which python` resolves to
    `/home/odbadmin/.pyenv/versions/py311/bin/python`, both with it and
    without it (the global `py311`).
  - The 2026-10-18 log shows the shim on `PATH`. Its cron mail shows no
    pyenv or activation error.
  - The run does not log `sys.executable`, so the interpreter is resolved
    rather than recorded. Both pyenv branches lead to py311.
- **Correction: the cron does not restart MHW.**
  - Its last line, `pm2 restart mhwapi`, fails: `pm2` is only in
    `~/.npm-global/bin`, which is not on the cron's `PATH`.
  - The cron mail shows `pm2: command not found` from that line in the
    July, August, September, and October 2026 runs. PM2's log shows no
    `mhwapi` restart at those times.
  - Fixing it is the MHW maintainer's decision and is not part of this
    maintenance.
  - The cron still writes `data/mhw.zarr` with py311's Zarr at 04:50 on the
    18th, so the window must not overlap it.
- **Startup dependencies.**
  - The launcher exits 1 if any of the three `mhw-client` Dask TLS files is
    unreadable. `odbadmin` can read all three.
  - The app opens `data/mhw.zarr` (consolidated) and connects to the Dask
    scheduler. If the connection fails, it logs `Dask scheduler unavailable;
    falling back to local execution.` and still serves.
  - Output and errors are merged into `API/tmp/err.log`.
- **Reload and restart mechanics.**
  - Both APIs run Gunicorn 23.0.0 with `--reload`. py311 has no `inotify`
    package, so Gunicorn uses its polling reloader, which checks the loaded
    module files, Zarr's included, about once a second. Workers restart
    after the Zarr files change.
  - PM2 5.4.2 increments `restart_time` only when it restarts an online app
    (`lib/God/ActionMethods.js`). `pm2_restart_check.sh` therefore cannot
    verify starting a stopped app.
- **Other py311 processes:** the Dask scheduler (PID 96637) and worker (PID
  101867 and its children), without `--reload`. They are the PM2 apps
  `dask-scheduler` and `dask-worker`, both with `autorestart` true. Nothing
  else runs on py311.

### Dask and Zarr evidence (read-only, 2026-10-28)

The Dask scheduler and worker run on py311 without `--reload`, so a package
swap alone does not restart them.

Source evidence (MHW at `920f0da` with no local changes; the Tide release
directory):

- MHW opens `data/mhw.zarr` with `chunks='auto'` and registers its Dask
  client with `set_as_default=True`. Every `/api/mhw` and `/api/mhw/csv`
  request calls `.load()` on that data. With the client connected, the read
  runs as a Dask graph:
  - the scheduler deserializes it (`update_graph` in distributed 2025.3.0);
  - the worker runs the `getitem` tasks, which read the chunks through Zarr.

  Without a client, MHW reads in its own process.
- Tide also creates a default client in each worker, but it opens its store
  with `chunks=None` and submits nothing.
- `update_mhw.py` creates no Dask client.

Snapshot at 2026-10-28T15:19:58Z. It shows one moment, not what will arrive
during a later window.

- The scheduler had 4 clients. `ss` shows they are the two MHW and two Tide
  Gunicorn workers; no other process was connected. There was one worker,
  idle, and 0 tasks in any state (`/json/counts.json` and `/metrics` on
  `127.0.0.1:8787`).
- Since the scheduler started on 2026-10-24, nearly all of its task compute
  time (about 406 s) is in `open_dataset-…-getitem` prefixes named after the
  MHW variables.
- numcodecs' extension modules are loaded in the scheduler, the Dask worker
  process, and an MHW and a Tide worker (`/proc/<pid>/maps`).
  - Zarr imports numcodecs, so Zarr is probably loaded in each. This is an
    inference: pure-Python modules do not appear in `maps`.
  - Zarr 2.18.6 also imports some modules inside functions. Whether a data
    request reaches one that is not yet loaded was not determined.
- The worker runs under a nanny with an 8 GB memory limit. A restart by the
  nanny or by PM2 starts a new process; whether that start imports Zarr was
  not checked.
- MHW data requests in the nginx log:
  - 26 October: 369, of which 362 came from one client in one hour;
  - 27 October: 239, of which 232 came in one hour;
  - 28 October up to 23:12: 3.

  Bursts arrive at different hours.

What this means for the service order:

- **APIs running during `apply`:** an MHW data request during the swap makes
  MHW submit a Dask graph with Zarr-backed reads. The scheduler deserializes
  and schedules it, and the Dask worker runs the read tasks. Whether any of
  these processes loads a not-yet-loaded Zarr module during the swap is
  unknown. Requests can arrive at any hour, sometimes hundreds an hour.
- **Only the APIs stopped:** MHW is the only task submitter found in the
  source and in the snapshot, so stopping it removes the known source of
  Zarr work on Dask. The scheduler and worker would still run on py311, any
  process with a valid client certificate could connect, and a scheduler or
  worker restart during the swap is not covered. Stopping the APIs does not
  remove the import risk.
- Hence the stop-first proposal below stops the Dask worker and scheduler
  as well.

### Service handling: stop first (proposed, not approved)

Proposed order, not validated on VM24: stop every py311 service, check that
no process uses py311, run `apply`, check the package, then start Dask, Tide,
and MHW and run their checks. Both APIs are down from the first stop to the
last start. The rehearsal below ran this sequence against real PM2 and Dask
with dummy apps; it has never run in production.

The owner has said that, if quiescing is needed, Tide, MHW, the Dask worker,
and the scheduler this runbook needs would be stopped for the maintenance and
restarted afterwards. That is the owner's stated intent. It is not yet the
approval of this exact order, and it is not the MHW maintainer's agreement or
a confirmation from other Dask users: those stay open in the run sheet's
checklist.

| PM2 app | Stopped | Why |
|---|---|---|
| `mhwapi` | first | Every data request submits a Dask graph with Zarr-backed reads (see "Dask and Zarr evidence"), and its Gunicorn runs with `--reload`. |
| `tide` | second | It submits no Dask work, but it runs on the same py311 with Gunicorn `--reload`, whose polling reloader watches the loaded Zarr files. A reload during the swap can import an incomplete package. Nothing shows otherwise, so it is stopped. |
| `dask-worker` | after the drain check | It runs the Zarr-backed read tasks. It stops only after the scheduler reports no client and no task. |
| `dask-scheduler` | last | See below. |

**Decision: stop the Dask scheduler too.** Evidence and reasoning:

- It deserializes every MHW graph (`update_graph` in distributed 2025.3.0),
  and numcodecs, which Zarr imports, is loaded in it. Once both APIs and the
  worker are stopped and the drain check has passed, it has no known work.
- Stopped, the whole precondition for `apply` becomes one checkable fact:
  no process runs or maps a file from py311 (`py311_quiescent`). Running,
  that check would have to exempt it, and any process holding a Dask client
  certificate could still connect and submit a graph during the swap.
- Running, it would keep its pre-`apply` Zarr modules in memory while the
  restarted worker and APIs load 2.18.7, a mix that lasts until it is next
  restarted.
- The drain check requires zero clients, so stopping it disconnects nobody
  that was observed.
- Cost and residual risk of stopping it: its start has never run in
  production. If it fails, the start sequence stops before the APIs (see
  "Recovery"); Tide does not need Dask, and MHW falls back to local
  execution, which is a behaviour change that needs the MHW maintainer.
- Residual risk if it were kept running instead: the unobserved-client case
  above and the lasting version mix. This is why the proposal stops it.

### PM2 and Dask lifecycle facts

Source-derived from PM2 5.4.2 (the npm package, `lib/`) and distributed
2025.3.0, and exercised in the rehearsal unless marked otherwise:

- **`pre_stop` never runs.** PM2 5.4.2 has no such option: it is absent from
  `lib/API/schema.json` and referenced nowhere in `lib/`. The broad
  `pre_stop` kill pattern in `conf/ecosystem.config.js` is therefore never
  executed by `pm2 stop tide`. In the rehearsal, a `tide` app whose
  `pre_stop` would touch a marker file never created it, and a process named
  `tide_app` outside PM2 survived the stop. The live `tide` entry was not
  re-read this round; because PM2 ignores the key either way, preflight B
  instead requires the daemon to be `PM2 v5.4.2`. If it is not, stop: check
  `pre_stop` support for that version before any `pm2 stop`.
- **`pm2 stop NAME`** sends SIGINT to every process of the app's tree,
  children first, and SIGKILL to the tree if the main PID is still alive
  after `kill_timeout` (1600 ms by default). It waits only for the main PID.
  A stopped app is never restarted by `autorestart`. NAME is matched as an
  exact app name first and as a namespace only when no app has that name.
- **Starting a stopped app**: `pm2 restart NAME` starts it with its stored
  environment (no `--update-env`, so the operator's shell never reaches it)
  and leaves the restart count unchanged. `pm2 start NAME` is not used: when
  no app or namespace has that name, it falls through to starting a new app
  from a script path.
- **The Dask nanny**: the `dask-worker` command closes its nanny on SIGINT
  (`Closing Nanny … Reason: signal-2`), and the nanny closes the worker. The
  nanny restarts a worker only when the worker exits while the nanny itself
  is still running. In the rehearsal the worker log shows the close and no
  `Restarting worker`, and no process was left.
- On VM24, `dask-scheduler` and `dask-worker` are PM2 apps run through
  `/bin/bash` with interpreter `none`; PM2's PID is the scheduler or the
  nanny itself (read-only, 2026-10-28). The worker process and the
  multiprocessing resource tracker are the nanny's children.

### Helper scripts

All four are read-only except the two PM2 actions of the lifecycle script,
and each exits non-zero unless every check passes.

- `conf/pm2_app_lifecycle.sh stop|start|status|pid` stops or starts one PM2
  app and verifies it:
  - it calls pm2 only after proving the production daemon alive (the same
    check as `pm2_restart_check.sh`) and never follows an inherited
    `PM2_HOME`;
  - stop records the app's process tree, waits for every PID of it, then
    requires the app to stay stopped for `--settle` seconds with no process
    carrying the app's `--match` word (no respawn by PM2 or a nanny, no
    orphan);
  - start accepts only a stopped or errored app with no matching process,
    uses `pm2 restart`, and requires it online, stable for `--settle`
    seconds with an unchanged restart count, with `--expect-exec` in the
    PM2 PID's command line and exactly one matching process tree;
  - both require every other PM2 app to keep its status and PID;
  - every mode reads all PM2 apps and accepts only these status and PID
    pairs: `online` with a positive integer PID (otherwise its executable
    and process tree could not be checked); `stopped` with PID 0, or with
    no or a null `pid`, which PM2 5.4.2 gives an app it loaded as stopped
    from its dump (VM24's `odbbathy`); `errored` with PID 0. Any other pair,
    an unknown status, or a malformed entry fails the read before any pm2
    action. `launching` and `stopping` are accepted only for the target app,
    only while polling for the script's own `pm2 stop` or `pm2 restart` of
    it to finish, and the result is then checked against the strict list.
    Every other app is held to the strict list at every read, including
    during that polling.
- `conf/dask_scheduler_check.sh drained|ready` reads the scheduler's own
  `/health`, `/metrics`, and `/json/identity.json` on `127.0.0.1:8787`. It
  never creates a Dask client. `drained` needs zero clients and zero tasks
  in every state; `ready` needs N running workers, optionally each with a
  nanny port owned by a given PID and an exact client count.
- `conf/py311_quiescent.sh` passes only when every live process was
  inspected and none has the py311 prefix in its command line or in a
  mapped file. A live process whose command line or maps cannot be read
  fails the check; only a process whose `/proc` entry has vanished (it
  exited) is skipped. `--inspectable-only` checks just the access, as a
  preflight; only the full check, without the flag, looks for py311 use.
  It needs root, and root runs it only as the root-owned, hash-checked copy
  made by `root_quiescent` (run sheet, step 4), never from the checkout.
- `conf/py311_zarr_state.sh` prints `STATE before`, `STATE after`, or
  `STATE unknown` with reasons. A known state needs the Zarr version, the
  freeze (the `prepare` snapshot, or it with only the zarr line changed),
  `pip check`, and the installed zarr files (every RECORD hash, no extra
  file in the package) to agree. `--expect` makes it fail on the other
  known state.
- `conf/pm2_restart_check.sh --daemon-only` is used by preflight B only: it
  proves the production PM2 daemon alive and never calls pm2.

The run sheet depends on seven scripts of the approved commit: these five,
`conf/py311_zarr_align.sh`, and `conf/https_smoke.sh`. Its step 1 pins the
sha256 of all seven.

### What is tested, derived, and unknown

Tested:

- Unit tests with a simulated PM2 daemon and process table
  (`tests/test_stop_first_scripts.py`): stop and start pass, and fail
  closed on a respawn, an orphan, a surviving tree process, a duplicate
  tree, the wrong executable, a crash loop, a raised restart count, a failed
  pm2 call, a change to another app, and a dead daemon. A stopped app
  without a `pid` reads as PID 0 and can be started. Every pair outside the
  accepted list, as target or bystander, fails every mode with only `jlist`
  called: online without a positive integer PID, stopped or errored with a
  non-zero or non-integer PID, errored without one, `launching` or
  `stopping` with any PID including 0, and unknown statuses. A `stopping`
  or `launching` read of the target right after the script's own action is
  waited through; an unknown status there fails at once, and so does
  another app in `launching` or `stopping` during that polling, whether or
  not the target is in a transition too. The scheduler checks
  fail closed on clients, tasks, worker count, worker status, nanny owner,
  and an unreachable scheduler.
- Sandbox runs of the run sheet's own blocks (`tests/test_deployment_doc.py`)
  with the scripts replaced by stubs: preflight B and its refusals, the
  per-app match words, the stop and start order, a stop or start that halts
  at the first failure inside an `&&` list, skipped apps already in the
  target state, the quiescence step halting the stop on a py311 process or
  on a process it cannot read, and the recovery chains, which start nothing
  on `STATE unknown`.
- Sandbox runs of `root_quiescent` from the preflight D block
  (`tests/test_deployment_doc.py`, not as root: `sudo` dropped, `id`,
  `stat`, `mktemp`, `install`, and `sha256sum` replaced by stand-ins, the
  copy in a temporary directory instead of `/root`, the checker replaced by
  a stub). The copy's directory is removed after a passing check, a
  failing check, any other exit status, and a hash mismatch, which never
  runs the copy; each keeps its exit status. A cleanup that fails warns and
  keeps the status. A non-root uid, or a `/usr/bin/python3` owned by
  another user or writable by group or other, stops before anything is
  copied; any argument other than `--inspectable-only` returns 2 before
  that. Without the flag the copy runs the full check.
- Step 6 in the same sandbox: `stop_services` with the real
  `root_quiescent` runs the full check on the copy after the four stops
  and passes; a swapped checker (never run), a failing check, or a
  non-root uid fails it after the stops, without `OK services stopped` and
  without reaching `apply`; with `root_quiescent` undefined, or without the
  MHW baseline, it stops nothing.
- Steps 1 and 7 in the same sandbox: the step 1 check passes on a clean
  checkout at the commit and refuses a missing or wrong `SHA`, local
  changes, and a committed script that differs from its pinned hash; when
  the root check fails, step 7's `apply` does not start.
- Step 9's log check, in the same sandbox and with real GNU tools in an
  offline container (Debian 12, 2026-10-30): it prints `OK` for new lines
  without the message, and for the message only before the mark. It exits
  non-zero, without `OK`, when the message is in the new lines, the log is
  missing or unreadable, `tail` fails, the mark is missing, or the log was
  rotated, replaced, truncated, or truncated and regrown past the mark. It
  has not run on VM24.
- The same `root_quiescent`, with only `sudo` dropped, ran as root in a
  throwaway offline container (Debian 12, GNU coreutils, a root-owned
  `/usr/bin/python3`, 2026-10-30): with `--inspectable-only`,
  `OK inspectable` and exit 0; without it, `OK quiescent` and exit 0, and
  exit 1 naming a live process run from the py311 path; a tampered
  checkout file `FAILED` and was not run; `--all` returned 2; a
  `/usr/bin/python3` owned by another user or group-writable stopped it
  before the copy; `/root` was left empty each time. On VM24, as root,
  only `root_quiescent --inspectable-only` has run (preflight D,
  2026-10-30); the full check has not.
- `conf/py311_quiescent.sh` against a fake `/proc`: a py311 command line, a
  mapped py311 library from a process started through the pyenv shim, a
  live process whose maps cannot be read (fails closed), and a process that
  exited between listing and reading (skipped).
- The state check against a real venv with real pip and hand-built zarr
  wheels: before, after, a changed file, a missing file, a stray file, an
  extra package, and a Zarr that does not import. The same venv shows that
  without `--force-reinstall` pip reports a damaged 2.18.6 with intact
  metadata as "already satisfied"; `rollback` therefore force-reinstalls.
- `dev_tpxo10/rehearsal/stop_first_rehearsal.sh`, 2026-10-29, on macOS with
  PM2 5.4.2, distributed and dask 2025.3.0, Gunicorn 23.0.0, numpy 2.2.4,
  and Python 3.11 in an isolated `PM2_HOME`:
  - four apps shaped like VM24's (scheduler and nanny-managed worker run
    through `bash -c exec`, a Tide-like app whose PM2 PID is the Gunicorn
    master, an MHW-like app under a bash launcher whose requests compute
    through the default Dask client), under continuous MHW-like traffic;
  - stop MHW, stop Tide, `drained`, stop the worker, stop the scheduler, no
    process left from the venv; `pre_stop` never ran and the `tide_app`
    bystander survived; the nanny logged no worker restart;
  - start the scheduler, the worker (registered, nanny port owned by PM2's
    PID), Tide, and MHW; four clients again and both apps answering;
  - a scheduler that cannot bind its port crash-looped into `errored` and
    the start check failed; the stop check then brought it to `stopped`;
  - an extra Dask client made `drained` fail until it disconnected;
  - stopping an already stopped app was refused.
- Real PM2 5.4.2 in an isolated `PM2_HOME` (macOS, 2026-10-29): an app
  stopped, then `pm2 save`, `kill`, and `resurrect`, comes back with no
  `pid` in `jlist`, as on VM24. The `364af40` lifecycle script failed on it
  even when asked about another app; the current one reads it as
  `stopped 0`, starts it with `pm2 restart`, and stops it again. The full
  rehearsal above passed again with the current lifecycle script.
- Read-only on VM24, 2026-10-29 (see "VM24 preflight record"): preflights A
  and C, preflight B with the parser fixed, and the pre-stop checks of the
  four apps (executable and one match root each), cut off before any pm2
  action.
- Read-only on VM24, 2026-10-30 (see "VM24 preflight D record"): the
  `ea22a03` checkout and its hashes, preflights A and C, and the owner's
  preflight D as root, which checks access only.

Derived from source, not run against the real services:

- the PM2 and nanny behaviour above, for the VM24 entries;
- that the real Tide and MHW reconnect to a restarted scheduler at startup
  (both create their client at startup, and MHW again per request);
- `py311_quiescent.sh` on a real Linux `/proc`, run as root: macOS has no
  `/proc`, so the rehearsal used `pgrep` on the venv path and the unit tests
  a fake `/proc`. As odbadmin on VM24 it fails closed as designed (147
  processes of other users unreadable, 2026-10-29). As root on VM24 only
  the access check has run (preflight D, `--inspectable-only`, 491
  processes readable, 2026-10-30); the full check, which looks for py311
  use, has not run on VM24 and runs in the window after the stops. The
  nanny-port lookup with `ss` did run on VM24 (preflight B);
- every branch of "Recovery" on VM24: none has run there (see the status at
  the top of that section).

Unknown until the window:

- how long each real stop and start takes (the rehearsal's took about 10 to
  20 seconds each, including an 8-second stability wait; the run sheet uses
  the 10-second default);
- whether the real `dask-scheduler` and `dask-worker` PM2 entries start
  cleanly with their stored TLS arguments and the production certificates.
  The rehearsal used plain `tcp://`; the isolated VM24 TLS test (see "VM24
  preflight record") used production's flags with a throwaway CA, on other
  ports, outside PM2. Neither proves the production restart: this is a risk
  to accept before scheduling, or to narrow with another isolated test on
  other ports and its own `DASK_TEMPORARY_DIRECTORY`;
- whether any process outside PM2 uses py311 in the window without loading
  a compiled module from it (`py311_quiescent` sees command lines and
  mapped files only).

### VM24 preflight record (read-only, 2026-10-29)

Nothing was installed, no pm2 action ran, and no service was stopped or
restarted. `pm2 jlist` was the only PM2 command.

- Checkout `/home/odbadmin/tmp/tide-zarr-364af40`, detached at `364af40`
  with no tracked or untracked changes, cloned from a git bundle copied
  from a workstation (VM24 has no credentials for the repository). The
  parser fix below changes `pm2_app_lifecycle.sh`, so the approved commit
  needs a new checkout and new hashes.
- Preflights A and C, run verbatim from the run sheet: `prepared` present,
  no `applied`, both wheels match, zarr 2.18.6, freeze identical to the
  `prepare` snapshot, `STATE before`. Nothing in `$D` changed.
- Preflight B failed at `pm2_app_lifecycle.sh status`: the stopped app
  `odbbathy` has no `pid` in `jlist`, and the parser raised `KeyError`.
  With that line fixed in a separate copy it passed: PM2 5.4.2 daemon PID
  3352, MHW launcher PID 96946, Dask TLS files readable, no MHW update, the
  four apps online, and Dask with 4 clients, 0 tasks, and one worker whose
  nanny port PID 101867 owns (`ss`).
- The same copy, cut off before any pm2 action, passed the pre-stop checks
  of all four apps: the PM2 PID runs the expected executable, and exactly
  one match root lies in its tree.
- Of odbadmin's processes, only the four apps use py311 (10 processes).
  PM2 runs five other apps (`ghrsst`, `ghrsst_mcp`, `woa23`, `gateway`, and
  the stopped `odbbathy`); none uses py311. Other users' processes need the
  root check.
- `py311_quiescent.sh --inspectable-only` as odbadmin fails closed as
  designed (147 processes of other users unreadable). The root run was not
  done; run it against the approved checkout and record the script hash.
- An isolated Dask TLS test, with production's scheduler and worker flags
  on ports 18786 and 18787, a throwaway CA, `env -i` with its own `HOME`
  and `DASK_CONFIG`, py311 with `-I -B`, outside PM2:
  - a nanny-managed worker registered, a task round trip and a small
    dask.array computation succeeded, and `ready` and `drained` passed
    against it;
  - the scheduler refused plaintext, a client without a certificate
    (`CERTIFICATE_REQUIRED`), and a certificate from another CA
    (`UNKNOWN_CA`);
  - on SIGINT to the tree, children first, the nanny closed without
    restarting the worker, and the scheduler exited;
  - production showed the same PIDs, 4 clients, 1 worker, and 0 tasks
    before and after, and no test process was left.

  Its limits: it proves neither the production certificates nor a start
  under PM2. Its scheduler also used production's `/tmp/dask-scratch-space`,
  because `DASK_TEMPORARY_DIRECTORY` was not set, and rewrote the
  modification times of the lock files there (`locket` opens them with
  `"wb"`). Dask's leftover purge removes only directories whose `flock` it
  can take; production holds both of its own, and nothing was removed. A
  repeat must set its own `DASK_TEMPORARY_DIRECTORY`.
- distributed 2025.3.0 answered `/json/identity.json` once with HTTP 500
  right after the test's tasks ran (tuple keys in a worker's metrics). The
  Dask check read that as not ready, retried, and passed, but printed a
  Python traceback; an operator may see one in the window.
- The production Dask certificates are valid until 2027-11-25 (the CA until
  2036-10-21), and odbadmin can read the four keys. Only the expiry dates
  were read.

### VM24 preflight D record (read-only, 2026-10-30)

Nothing was installed or applied, no pm2 command ran, and no service was
stopped or restarted.

- Checkout `/home/odbadmin/tmp/tide-zarr-ea22a03`, detached at `ea22a03`
  (the merge of PR #5), built from a git bundle by the operator: clean, no
  tracked or untracked changes, `git fsck` clean. The six run-sheet
  scripts match the commit's blobs and sha256 values; the pinned `H` of
  preflight D equals the sha256 of `conf/py311_quiescent.sh`
  (`12af24d65965ad1e80ef884c8530d3266c3bc1a95f1260feaa3f578ab6fc763f`).
  Preflights A and C passed again (15:04 CST): `prepared` present, no
  `applied`, both wheels match, zarr 2.18.6, freeze identical to the
  `prepare` snapshot, `STATE before`.
- Preflight D, run by the owner as root, 2026-10-30 15:17:38 CST (VM24
  local time, Asia/Taipei): PASS, exit 0. The root-owned copy matched the
  pinned hash (`q.sh: OK`), and `--inspectable-only` read all 491 live
  processes. No `WARN` was printed, so the trap removed the copy and its
  directory. Output:

  ```text
  root:root 700 /root/odbz-quiescent.E2c1M5
  root:root 500 /root/odbz-quiescent.E2c1M5/q.sh
  /root/odbz-quiescent.E2c1M5/q.sh: OK
  OK inspectable: all 491 live processes could be read
  exit 0
  ```

- What it shows: root can read the command line and memory maps of every
  live process, so the full check can run at all. What it does not show:
  that no process used py311. `--inspectable-only` ignores py311 use, and
  the services were running. The full quiescence check (step 6,
  `root_quiescent` without the flag, inside `stop_services`) must still
  pass in the window after every service has stopped.
- This does not make `apply` runnable and is not a deployment. The open
  items of the run sheet's checklist must all be checked first.

### MHW post-start smoke (proposed)

MHW is started by `app start mhwapi` (step 8 of the run sheet), which runs
`pm2_app_lifecycle.sh start` with `--expect-exec
/home/odbadmin/proj/marineheatwave/API/conf/start_app.sh`: the launcher is
PM2's PID for `mhwapi`. It has never run against the real `mhwapi`; the
rehearsal used a dummy app under a bash launcher.

Then `mhw_check` (defined in the run sheet) runs these read-only checks:

| Check | Expected |
|---|---|
| Listener on `127.0.0.1:8030` (`ss`, `ps`) | Exactly one Gunicorn master whose parent is the MHW launcher. Its command line starts with `py311/bin/python3.11 py311/bin/gunicorn mhw_app:app`. |
| GET `https://eco.odb.ntu.edu.tw/api/swagger/mhw/openapi.json` | 200 `application/json`. `paths` include `/api/mhw` and `/api/mhw/csv`. |
| GET `https://eco.odb.ntu.edu.tw/api/swagger/mhw` | 200 `text/html; charset=utf-8`. The page references `/api/swagger/mhw/openapi.json`. |
| GET `https://eco.odb.ntu.edu.tw/api/mhw?lon0=135&lat0=15&start=2024-01-01&end=2024-03-01` | 200 `application/json`. Three rows at lon 135.125, lat 15.125, dated 2024-01-01, 2024-02-01, and 2024-03-01, each with `level` 0–4. The body matches the pre-apply baseline byte for byte. |
| GET `https://eco.odb.ntu.edu.tw/api/mhw` (no parameters) | 422 `application/json`, reporting `lon0` and `lat0` missing. |

- It does not call `/api/mhw/csv`, which writes a temporary file for each
  request, `/api/mhw/plot`, which is heavy, or anything that updates data.
- After `mhw_check`, look at the new lines of `API/tmp/err.log` for
  `Dask scheduler unavailable`. MHW still serves without Dask, so report it
  rather than fail.
- Read-only baseline, 2026-10-28:
  - The four GETs, from outside VM24, returned 200, 200, 200, and 422 as
    above. The point query was 178 bytes, sha256 `23ea145e…`, with levels
    1, 0, and 0.
  - `mhw_proc_check` on VM24 printed
    `OK mhw process: master 96947 is the py311 Gunicorn under the launcher`.
  - A second `mhw_http_check` matched the baseline byte for byte.
- The MHW maintainer should confirm these expectations.

### Procedure

`conf/py311_zarr_align.sh` runs under `set -euo pipefail`, independent of
the operator's shell. Every step either succeeds or exits non-zero before
the next one. A failed `pip install` is not hidden by the `tee` that logs
it, and a failed snapshot leaves no freeze file behind. Nothing chained
after it with `&&` runs on failure.

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

`prepare` records the sorted `pip freeze --all`, the accepted `pip check`
result with its exit status, and the Zarr version, changing nothing in
py311. It downloads the 2.18.7 and 2.18.6 wheels pinned by SHA-256 into
`$D/wheels` and dry-runs the install, which must offer exactly
`zarr-2.18.7`. It was run on 2026-10-28 (see the status above). Do not run
it again for this window: the run sheet reuses that directory and its
wheels. Run it only for a new attempt after a rollback, into a new
directory, with `SRC` and `ALIGN` set as in step 1 of the run sheet:

```bash
D=/home/odbadmin/tmp/zarr-align-$(date +%Y%m%d)   # must not exist yet
"$ALIGN" prepare --dir "$D"
```

`apply` refuses to start unless `$D` was prepared and not yet applied, and
unless py311 still matches the `prepare` freeze. It installs 2.18.7 offline
with `--no-deps`, `--no-index`, and `--require-hashes`. Then it requires:

- the freeze to differ only in `zarr==2.18.6` → `zarr==2.18.7`;
- an accepted `pip check` result identical to the `prepare` one;
- zarr to import as 2.18.7.

`rollback` installs the saved 2.18.6 wheel the same way, with
`--force-reinstall`: without it, pip reports a damaged 2.18.6 whose metadata
is intact as already satisfied and rewrites nothing (shown with real pip in
`tests/test_stop_first_scripts.py`). It requires the freeze and an accepted
`pip check` result to match the `prepare` snapshot exactly. A directory
accepts one `apply`.

### Production run sheet

Checklist. Every sign-off gate must be checked, with its answer recorded
here, before the window starts; every in-window gate is checked as it
passes. Do not run `apply` while any item is open.

The owner's Zarr-only authorization does not settle any of them. In
particular, it does not approve the stop-first order, which is still a
proposal.

Done, with the evidence recorded:

- [x] Owner authorization, 2026-10-28: Zarr 2.18.6 → 2.18.7 in the shared
      py311 only.
- [x] Isolated validation (2026-10-27). The freeze changed only in the
      zarr line, the tests went green, and the MHW-style round trip passed.
- [x] `prepare` on VM24 (2026-10-28 10:34), in
      `/home/odbadmin/tmp/zarr-align-20261028T103416`:
      - `freeze-before.txt`, `pip-check-before.txt` (four accepted
        conflicts), `dry-run.txt` (`Would install zarr-2.18.7` only), the
        `prepared` marker, and no `applied` marker;
      - both wheels match their pinned hashes (rechecked later on
        2026-10-28).
- [x] Production PM2 daemon, read-only: PID 3352, title
      `PM2 v5.4.2: God Daemon (/home/odbadmin/.pm2)`.
- [x] MHW API and cron interpreters: pyenv resolves both to py311,
      read-only (see "MHW facts"). The launcher's TLS files are readable.
- [x] MHW smoke baseline taken (see "MHW post-start smoke").
- [x] Dask usage and snapshot, read-only (see "Dask and Zarr evidence").
- [x] Isolated stop-first rehearsal, 2026-10-29 (see "What is tested").
- [x] Owner, before the window: preflight D (step 4), the read-only root
      `--inspectable-only` check through the root-owned copy. PASS on
      2026-10-30 15:17:38 CST: the copy's hash line `q.sh: OK`, all 491
      live processes readable, exit 0 (see "VM24 preflight D record").
      This shows only that root can read every process; it does not show
      that no process used py311, and it does not make `apply` runnable.

Sign-off gates, each open until the person named answers:

- [ ] Owner and MHW maintainer: the window, with its date and time. It
      must not overlap the MHW cron at 04:50 on the 18th. Both APIs are
      down for the whole sequence. The owner will choose a window that does
      not conflict with the existing schedules; no window is set yet.
- [ ] Owner: approve the stop-first order, including stopping and starting
      the Dask scheduler and worker in the window, or choose another. The
      owner's stated intent is recorded under "Service handling"; it does
      not check this item.
- [ ] MHW maintainer: agree to stopping and starting `mhwapi`,
      `dask-worker`, and `dask-scheduler` through PM2 (never run against
      the real apps), and to the smoke expectations. Also say whether a
      restart appends to `API/tmp/err.log`: this is not recorded here, and
      if it truncates the log, step 9's log check cannot pass and goes to R4.
- [ ] Owner or the maintainer of the host Dask runbook: confirm that no
      service other than Tide and MHW needs the Dask scheduler in the
      window. Only those two were connected on 2026-10-28.
- [ ] Owner: accept the risk that the real `dask-scheduler` and
      `dask-worker` PM2 entries have never been restarted with the
      production certificates (see "Unknown until the window"), or ask for
      another isolated test first.
- [ ] Owner: reachable during the whole window for the recovery decisions
      below.
- [ ] Owner: name who runs the run sheet. It runs in one shell, and steps
      4, 6, and 7 need root in that shell, so it must be the owner or a
      root-capable operator the owner names.
- [ ] Reviewer: approve this run sheet and its scripts.
- [ ] Operator: a checkout `SRC` of the approved commit on VM24, with no
      tracked changes; step 1 repeats this check in the window. The sha256
      of the seven scripts pinned in step 1 must match that commit.
      `py311_quiescent.sh` runs as root, so its hash matters most. The
      2026-10-28 `prepare` directory stays valid: this change to the
      aligner touches only `rollback`. The `ea22a03` checkout passed these
      checks for six scripts on 2026-10-30 (see "VM24 preflight D record"),
      but any later approved commit needs its own new checkout.

In-window gates, checked in order as they pass. Both must be checked before
step 7:

- [ ] In the window: preflights A to D pass immediately before the
      stop. Preflight D here, like the record above, checks access only.
- [ ] Owner: root in the window for the full quiescence check of step 6
      (`root_quiescent` inside `stop_services`, run after every service
      stopped). It must read every process's memory maps; as odbadmin it
      fails closed while other users' processes run, and then no `apply`
      is possible. Only this check looks for py311 use, and it must print
      `OK quiescent`.

Only Zarr may change. In the window, run no `pip`, `pyenv`, or `uv`
command against py311 other than the ones below. Do not re-run `prepare`
into `$D`, download wheels again, or edit anything in `$D`.

Run the steps in order, in one shell on VM24, one step at a time. Start a
step only after the previous one printed its `OK` line, and never paste
several steps together. Each block stops at the first failure. On any
failure, follow "Recovery" and report to the owner; never skip ahead.

If the shell is lost, its variables and functions are gone. Run the step 1
blocks, the preflight D block of step 4, and the two definition blocks of
step 5 again; none of them changes anything, and preflight D also passes
with the services stopped. Then read the package state with
`"$ZSTATE" --dir "$D"` and continue from the step or "Recovery" branch that
state calls for. Do not repeat steps 2, 3, or 6 to find out where things
stand. The `err.log` mark of step 5 is lost with the shell, so step 9's log
check fails and R4 applies.

1. Set `SHA` to the first seven or more characters of the approved commit
   recorded in the checklist, then set the variables. `$D` is the
   2026-10-28 `prepare` directory.

   ```bash
   SRC=/home/odbadmin/tmp/tide-zarr-$SHA   # approved checkout, see checklist
   ALIGN="$SRC/conf/py311_zarr_align.sh"
   ZSTATE="$SRC/conf/py311_zarr_state.sh"
   DASKC="$SRC/conf/dask_scheduler_check.sh"
   D=/home/odbadmin/tmp/zarr-align-20261028T103416
   ```

   Then check the checkout, read-only: it is at the approved commit, clean,
   and all seven scripts the run sheet uses match their reviewed hashes.

```bash
(
  set -euo pipefail
  fail() { echo "FAIL preflight: $*; do not apply" >&2; exit 1; }
  sha=${SHA:-}
  [ "${#sha}" -ge 7 ] || fail "set SHA to the approved commit first"
  case "$(git -C "$SRC" rev-parse HEAD)" in
    "$sha"*) ;;
    *) fail "$SRC is not at $sha" ;;
  esac
  [ -z "$(git -C "$SRC" status --porcelain)" ] || fail "$SRC has local changes"
  cd "$SRC"
  sha256sum -c - <<'SUMS' || fail "a script does not match its reviewed hash"
a159c31e7c62eb939e0ff9038b79aea9969c5c905680adfcc7a7f7757c204da9  conf/py311_zarr_align.sh
d0edc2c2782556a1f1c3c619ae6f59f49bacf8ec3161f3afc3ec4ae6d38a8018  conf/py311_zarr_state.sh
12af24d65965ad1e80ef884c8530d3266c3bc1a95f1260feaa3f578ab6fc763f  conf/py311_quiescent.sh
961d8ef7f0bf0da955b004015664ea737c77226478663f415ddb3bd73e26044d  conf/pm2_app_lifecycle.sh
cde39dccce54eeb9a962b4a8e67f3afae569a94681db19b2c3253c3ab498bab3  conf/dask_scheduler_check.sh
3d096450da7f363efe4b259bfb0b18a399fe31a8f48bf0c43ac7043654ed4730  conf/https_smoke.sh
304a4f66edead6b86e0670ba54c7476dacc27ef4847cce2183929e6bbc969cc9  conf/pm2_restart_check.sh
SUMS
  echo "OK checkout: $SRC at $sha, clean, seven scripts match"
)
```

2. Preflight A, read-only: the artifacts are intact, and py311 still
   matches the `prepare` snapshot. `apply` repeats the freeze comparison,
   but this block catches drift before any service stops.

```bash
(
  set -euo pipefail
  PY=/home/odbadmin/.pyenv/versions/py311/bin/python
  fail() { echo "FAIL preflight: $*; do not apply" >&2; exit 1; }
  [ -f "$D/prepared" ] || fail "$D has no prepared marker"
  [ ! -e "$D/applied" ] || fail "$D was already used for apply"
  cd "$D"
  sha256sum -c - <<'SUMS' || fail "a saved wheel does not match its pinned hash"
ac3dc4033e9ae4e9d7b5e27c97ea3eaf1003cc0a07f010bd83d5134bf8c4b223  wheels/zarr-2.18.7-py3-none-any.whl
a5334311fae8859870d78561ebb6bd0ab01accc392a6cf4dcaa25d83b4a2f22c  wheels/zarr-2.18.6-py3-none-any.whl
SUMS
  [ "$("$PY" -I -c 'import zarr; print(zarr.__version__)')" = 2.18.6 ] \
    || fail "zarr does not import as 2.18.6"
  now=$(mktemp)
  "$PY" -I -m pip --disable-pip-version-check freeze --all | LC_ALL=C sort -f > "$now" \
    || fail "pip freeze failed"
  diff "$D/freeze-before.txt" "$now" \
    || fail "py311 changed since prepare (current freeze kept in $now)"
  rm "$now"
  echo "OK preflight A: zarr 2.18.6; freeze identical to the prepare snapshot"
)
```

3. Preflight B, read-only: the services are as they were checked on
   2026-10-28.
   - If the MHW launcher is no longer PID 96946, PM2 has started it again
     since then. Resolve the interpreter again (see "MHW facts") before
     continuing.
   - The MHW update must not be running.
   - The PM2 daemon must be 5.4.2, whose `pm2 stop` never runs `pre_stop`.
   - All four apps are online, and Dask has one worker under the PM2 nanny
     and the four API clients. Anything else is an unknown state: stop.

```bash
(
  set -euo pipefail
  fail() { echo "FAIL preflight: $*; do not apply" >&2; exit 1; }
  "$SRC/conf/pm2_restart_check.sh" --daemon-only --pm2-home /home/odbadmin/.pm2
  title=$(ps -o args= -p "$(tr -dc '0-9' < /home/odbadmin/.pm2/pm2.pid)")
  [ "$title" = "PM2 v5.4.2: God Daemon (/home/odbadmin/.pm2)" ] \
    || fail "the PM2 daemon is '$title'; check pre_stop support for that version first"
  [ "$(ps -o args= -p 96946)" = "bash /home/odbadmin/proj/marineheatwave/API/conf/start_app.sh" ] \
    || fail "the MHW launcher is no longer PID 96946"
  for f in certs/ca.crt certs/mhw-client.crt private/mhw-client.key; do
    [ -r "/home/odbadmin/.config/dask/tls/$f" ] \
      || fail "the MHW launcher would exit: $f is unreadable"
  done
  if pgrep -u odbadmin -f 'update_mhw\.(sh|py)' > /dev/null; then
    fail "the MHW update is running"
  fi
  life="$SRC/conf/pm2_app_lifecycle.sh"
  for a in dask-scheduler dask-worker tide mhwapi; do
    s=$("$life" status --pm2-home /home/odbadmin/.pm2 --name "$a")
    [ "${s%% *}" = online ] || fail "$a is not online: $s"
  done
  nanny=$("$life" pid --pm2-home /home/odbadmin/.pm2 --name dask-worker)
  "$SRC/conf/dask_scheduler_check.sh" ready --workers 1 --nanny-pid "$nanny" \
    --clients 4 --timeout 10 \
    || fail "Dask is not one worker under the PM2 nanny with the four API clients"
  echo "OK preflight B: PM2 5.4.2 alive; MHW launcher unchanged; no MHW update; four apps online; Dask as expected"
)
```

4. Preflights C and D, read-only.
   - C: py311 is exactly in the `before` state, with every installed zarr
     file matching its RECORD.

   ```bash
   "$ZSTATE" --dir "$D" --expect before
   ```

   - D, run by the owner only, as root, read-only: the quiescence check can
     read the command line and memory maps of every live process. It
     ignores py311 use here, since the services still run; it proves the
     step-6 check can pass at all. Its `OK inspectable` is not `OK quiescent`:
     it says nothing about py311 use and never replaces step 6.
   - The block defines `root_quiescent [--inspectable-only]` and runs it
     with `--inspectable-only`. It stays defined in this shell: step 6's
     full check (`py311_quiescent`, inside `stop_services`) calls the same
     function without the flag, so both root runs go through the copy
     below. In the window, run this block again in the same shell before
     step 5; `stop_services` refuses to stop anything if `root_quiescent`
     is not defined.
   - The checkout is writable by odbadmin, and so by every service running
     as odbadmin, so root never runs `$SRC/conf/py311_quiescent.sh` in
     place: the file could be swapped between a hash check and the run.
     Root copies it into a new root-only directory under `/root` (never
     `/tmp`, whose owner is `nginx`), checks the copy's sha256, and runs
     that copy.
   - The root shell starts from an empty environment (`env -i`) with a
     fixed system `PATH`, and stops before copying anything unless it is
     uid 0 and `python3` resolves to `/usr/bin/python3`, owned by root and
     not writable by group or other. That Python runs the checker's own
     code and removes the copy and its directory afterwards (`os.unlink`,
     `os.rmdir`), so the block calls no `rm` or `rmdir`.
   - `root_quiescent` returns the check's own status: 0 only for
     `OK inspectable` (or, without the flag, `OK quiescent`); 1 for a hash
     mismatch (the copy never runs) or a failed check; 2 for any argument
     other than `--inspectable-only`. A failed cleanup prints a warning and
     keeps that status. Expected output here:
     `root:root 700 /root/odbz-quiescent.…`, `root:root 500 …/q.sh`,
     `…/q.sh: OK`, `OK inspectable: all N live processes could be read`,
     `exit 0`.

   ```bash
   # root_quiescent [--inspectable-only]: run the checkout's
   # py311_quiescent.sh as root only through a root-owned copy whose sha256
   # matches; returns the check's status. Step 6 uses it too.
   root_quiescent() {
     /usr/bin/sudo /usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin \
       /bin/bash --noprofile --norc -s -- "$SRC/conf/py311_quiescent.sh" "$@" <<'EOF'
   set -euo pipefail
   umask 077
   export PATH=/usr/sbin:/usr/bin:/sbin:/bin
   S=${1:?the checkout copy of py311_quiescent.sh}
   M=${2:-}
   [ "$#" = 1 ] || { [ "$#" = 2 ] && [ "$M" = --inspectable-only ]; } \
     || { echo "FAIL: arguments must be SCRIPT [--inspectable-only]" >&2; exit 2; }
   H=12af24d65965ad1e80ef884c8530d3266c3bc1a95f1260feaa3f578ab6fc763f
   [ "$(/usr/bin/id -u)" = 0 ] || { echo "FAIL: not uid 0" >&2; exit 1; }
   [ "$(command -v python3)" = /usr/bin/python3 ] \
     || { echo "FAIL: python3 does not resolve to /usr/bin/python3" >&2; exit 1; }
   read -r po pm < <(/usr/bin/stat -L -c '%U %a' /usr/bin/python3) || true
   [ "${po:-}" = root ] && [[ "${pm:-}" =~ ^[0-7]{3,4}$ ]] && (( (8#$pm & 8#022) == 0 )) \
     || { echo "FAIL: /usr/bin/python3 is '${po:-?}' mode '${pm:-?}', not root-owned and closed to group/other writes" >&2; exit 1; }
   T=$(/usr/bin/mktemp -d /root/odbz-quiescent.XXXXXX)
   cleanup() {
     /usr/bin/python3 -I -c 'import os, sys
   d = sys.argv[1]
   f = os.path.join(d, "q.sh")
   if os.path.lexists(f):
       os.unlink(f)
   os.rmdir(d)' "$T"
   }
   trap 'rc=$?; cleanup || echo "WARN: could not remove $T; remove it by hand" >&2; exit "$rc"' EXIT
   /usr/bin/install -m 0500 -o root -g root -- "$S" "$T/q.sh"
   /usr/bin/stat -c '%U:%G %a %n' "$T" "$T/q.sh"
   echo "$H  $T/q.sh" | /usr/bin/sha256sum -c -
   rc=0; /bin/bash --noprofile --norc "$T/q.sh" ${M:+"$M"} || rc=$?
   echo "exit $rc"
   exit "$rc"
   EOF
   }
   root_quiescent --inspectable-only
   ```

5. Define the MHW checks and the maintenance helpers in the same shell,
   then take the MHW baseline right before the stop, with the third block
   below. `stop_services` refuses to stop anything without that baseline,
   because every start compares against it. The same block marks where
   `err.log` stands, for step 9; it marks only after a new baseline, so
   running it again cannot move the mark.

```bash
# Read-only MHW checks. Nothing here calls pm2 or writes outside OUT.
#   mhw_proc_check             the 8030 listener is py311 Gunicorn under the launcher
#   mhw_http_check OUT [BASE]  four GETs through 443; responses saved in OUT
#   mhw_check OUT [BASE]       both
# OUT must not exist yet. With BASE, the point query must match BASE byte
# for byte.
mhw_proc_check() (
  set -euo pipefail
  launcher="bash /home/odbadmin/proj/marineheatwave/API/conf/start_app.sh"
  py311=/home/odbadmin/.pyenv/versions/py311/bin
  master=""
  for pid in $(ss -ltnpH 'sport = :8030' | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
    [ "$(ps -o args= -p "$(ps -o ppid= -p "$pid" | tr -d ' ')")" = "$launcher" ] || continue
    [ -z "$master" ] || { echo "FAIL mhw: two 8030 masters under the launcher" >&2; exit 1; }
    master=$pid
  done
  [ -n "$master" ] || { echo "FAIL mhw: no 8030 listener under the MHW launcher" >&2; exit 1; }
  args=$(ps -o args= -p "$master")
  case "$args" in
    "$py311/python3.11 $py311/gunicorn mhw_app:app "*) ;;
    *) echo "FAIL mhw: 8030 master $master runs '$args', not the py311 Gunicorn" >&2; exit 1 ;;
  esac
  echo "OK mhw process: master $master is the py311 Gunicorn under the launcher"
)
mhw_http_check() (
  set -euo pipefail
  out=${1:?usage: mhw_http_check OUT [BASE]}
  base=${2:-}
  mkdir "$out" || { echo "FAIL mhw: $out already exists or cannot be created" >&2; exit 1; }
  expect() {   # expect FILE PATH STATUS CONTENT_TYPE
    local got
    got=$(curl -sS --proto '=https' --max-time 60 -o "$out/$1" \
            -w '%{http_code} %{content_type}' "https://eco.odb.ntu.edu.tw$2") \
      || { echo "FAIL mhw: GET $2: curl failed" >&2; exit 1; }
    [ "$got" = "$3 $4" ] \
      || { echo "FAIL mhw: GET $2 gave '$got', expected '$3 $4'" >&2; exit 1; }
  }
  expect openapi.json /api/swagger/mhw/openapi.json 200 application/json
  expect swagger.html /api/swagger/mhw 200 'text/html; charset=utf-8'
  expect point.json '/api/mhw?lon0=135&lat0=15&start=2024-01-01&end=2024-03-01' 200 application/json
  expect missing.json /api/mhw 422 application/json
  python3 -I - "$out" <<'PY' || { echo "FAIL mhw: a response lacks an expected feature" >&2; exit 1; }
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
paths = json.loads((d / "openapi.json").read_text())["paths"]
assert {"/api/mhw", "/api/mhw/csv"} <= set(paths), sorted(paths)
assert "/api/swagger/mhw/openapi.json" in (d / "swagger.html").read_text()
rows = json.loads((d / "point.json").read_text())
assert [r["date"] for r in rows] == ["2024-01-01", "2024-02-01", "2024-03-01"], rows
for r in rows:
    assert (r["lon"], r["lat"]) == (135.125, 15.125), r
    assert r["level"] in (0, 1, 2, 3, 4), r
missing = {e["loc"][-1] for e in json.loads((d / "missing.json").read_text())["detail"]}
assert missing == {"lon0", "lat0"}, missing
PY
  if [ -n "$base" ]; then
    cmp "$base/point.json" "$out/point.json" \
      || { echo "FAIL mhw: point query differs from $base" >&2; exit 1; }
  fi
  echo "PASS mhw http: 4 GETs as expected${base:+; point query identical to $base}"
)
mhw_check() { mhw_proc_check && mhw_http_check "$@"; }
```

```bash
# Maintenance helpers. They use SRC, DASKC, and D from step 1, root_quiescent
# from step 4, and mhw_check from the block above.
#   app stop|start NAME   one PM2 app through pm2_app_lifecycle.sh
#   py311_quiescent       as root, through root_quiescent's hash-checked copy:
#                         no live process runs or maps a file from py311, and
#                         none could not be inspected
#   stop_services         MHW, Tide, drain, worker, scheduler, then quiescent
#   start_services SMOKE_OUT MHW_OUT
#                         scheduler, worker, Tide + smoke, MHW + smoke, Dask
# stop_services and start_services skip an app that is already in the target
# state, so they also serve the recovery branches. Every step ends in
# `|| exit 1`: bash ignores set -e inside a function called from an && list,
# as these are, so the checks are explicit.
PM2H=/home/odbadmin/.pm2
PY311=/home/odbadmin/.pyenv/versions/py311/bin
app() (
  case "$2" in
    tide) match=tide_app:app; exe=$PY311/gunicorn ;;
    mhwapi) match=mhw_app:app; exe=/home/odbadmin/proj/marineheatwave/API/conf/start_app.sh ;;
    dask-worker|dask-scheduler) match=$PY311/$2; exe=$match ;;
    *) echo "FAIL: unknown app '$2'" >&2; exit 2 ;;
  esac
  "$SRC/conf/pm2_app_lifecycle.sh" "$1" --pm2-home "$PM2H" --name "$2" \
    --match "$match" --expect-exec "$exe"
)
app_status() (
  s=$("$SRC/conf/pm2_app_lifecycle.sh" status --pm2-home "$PM2H" --name "$1") || exit 1
  echo "${s%% *}"
)
nanny_pid() { "$SRC/conf/pm2_app_lifecycle.sh" pid --pm2-home "$PM2H" --name dask-worker; }
# Reading other users' memory maps needs root; as odbadmin the check
# cannot pass while their processes run (see the checklist). Root never runs
# the odbadmin-writable checkout file in place: root_quiescent (step 4) runs
# a root-owned copy whose sha256 matches.
py311_quiescent() { root_quiescent; }
stop_services() (
  # Refuse before stopping anything if the final check could not run, or if
  # a later start would have no baseline to compare with.
  declare -F root_quiescent > /dev/null \
    || { echo "FAIL: root_quiescent is not defined; run step 4 in this shell first" >&2; exit 1; }
  [ -f "$D-mhw-before/point.json" ] \
    || { echo "FAIL: no MHW baseline in $D-mhw-before; run step 5 first" >&2; exit 1; }
  for a in mhwapi tide dask-worker dask-scheduler; do
    st=$(app_status "$a") || exit 1
    if [ "$a" = dask-worker ]; then
      sched=$(app_status dask-scheduler) || exit 1
      if [ "$sched" = online ]; then "$DASKC" drained || exit 1; fi
    fi
    if [ "$st" = stopped ]; then
      echo "note: $a already stopped"
    else
      app stop "$a" || exit 1
    fi
  done
  py311_quiescent || exit 1
  echo "OK services stopped: MHW, Tide, Dask worker, Dask scheduler"
)
start_services() (
  smoke_out=${1:?usage: start_services SMOKE_OUT MHW_OUT}
  mhw_out=${2:?usage: start_services SMOKE_OUT MHW_OUT}
  start_if_down() {
    local st
    st=$(app_status "$1") || return 1
    case "$st" in
      online) echo "note: $1 already online; not started" ;;
      stopped|errored) app start "$1" || return 1 ;;
      *) echo "FAIL: $1 is $st, an unexpected state" >&2; return 1 ;;
    esac
  }
  start_if_down dask-scheduler || exit 1
  st=$(app_status dask-worker) || exit 1
  if [ "$st" = online ]; then n=1; else n=0; fi
  "$DASKC" ready --workers "$n" || exit 1
  start_if_down dask-worker || exit 1
  nanny=$(nanny_pid) || exit 1
  "$DASKC" ready --workers 1 --nanny-pid "$nanny" || exit 1
  start_if_down tide || exit 1
  "$SRC/conf/https_smoke.sh" --host eco.odb.ntu.edu.tw --mode prod \
    --prod-port 443 --out-root "$smoke_out" || exit 1
  start_if_down mhwapi || exit 1
  mhw_check "$mhw_out" "$D-mhw-before" || exit 1
  nanny=$(nanny_pid) || exit 1
  "$DASKC" ready --workers 1 --nanny-pid "$nanny" --clients 4 || exit 1
  echo "OK services started: Dask, Tide, and MHW pass their checks"
)
```

```bash
MHW_LOG=/home/odbadmin/proj/marineheatwave/API/tmp/err.log
mhw_check "$D-mhw-before" \
  && mhw_log_id=$(stat -c '%d:%i' "$MHW_LOG") \
  && mhw_log_bytes=$(wc -c < "$MHW_LOG") \
  && mhw_log_head=$(head -c "$mhw_log_bytes" "$MHW_LOG" | sha256sum) \
  && echo "OK baseline: err.log marked at $mhw_log_bytes bytes"
```

6. Stop every py311 service. This must end with `OK quiescent` from the
   full root check, then `OK services stopped`. If it fails, nothing was
   installed: follow R0.

   ```bash
   stop_services
   ```

7. Apply, then require the `after` state. Run this only after every
   checklist item is checked and step 6 printed `OK services stopped`. The
   chain repeats the full root check right before `apply`, which has no
   quiescence check of its own, so `apply` cannot start while the check
   fails. If this fails, follow R1.

   ```bash
   root_quiescent && "$ALIGN" apply --dir "$D" && "$ZSTATE" --dir "$D" --expect after
   ```

8. Start the services against the validated 2.18.7, with all checks. Run
   this only after step 7 printed `OK apply` and `STATE after`. If it fails,
   follow R2.

   ```bash
   start_services "$D-smoke" "$D-mhw-after"
   ```

9. After `OK services started`, with the services up:
   - check the lines added to MHW's `err.log` since step 5 for the Dask
     fallback message, which should be absent now that the scheduler was up
     first. Only a complete read that finds no message prints `OK`. The
     block exits non-zero if it finds the message, if the log cannot be
     read, if the mark from step 5 is missing, or if the log was rotated,
     replaced, or truncated, because then the new lines cannot be told
     from the old ones:

     ```bash
     (
       set -euo pipefail
       fail() { echo "FAIL step 9: $*; follow R4" >&2; exit 1; }
       [ -n "${MHW_LOG:-}" ] && [ -n "${mhw_log_id:-}" ] \
         && [ -n "${mhw_log_bytes:-}" ] && [ -n "${mhw_log_head:-}" ] \
         || fail "no err.log mark from step 5 in this shell"
       same_log() {
         [ -f "$MHW_LOG" ] && [ -r "$MHW_LOG" ] || fail "cannot read $MHW_LOG"
         [ "$(stat -c '%d:%i' "$MHW_LOG")" = "$mhw_log_id" ] \
           || fail "$MHW_LOG was rotated or replaced since step 5"
         [ "$(head -c "$mhw_log_bytes" "$MHW_LOG" | sha256sum)" = "$mhw_log_head" ] \
           || fail "$MHW_LOG was truncated or rewritten since step 5"
       }
       same_log
       new=$(tail -c +"$((mhw_log_bytes + 1))" "$MHW_LOG") || fail "cannot read $MHW_LOG"
       same_log
       case "$new" in
         *'Dask scheduler unavailable'*) fail "Dask fallback in the new lines of err.log" ;;
       esac
       echo "OK no Dask fallback in the new lines of err.log"
     )
     ```

   - run the py311 test suite of the approved commit as in release sequence
     step 2: in a separate checkout with read-only store links, never in
     `$SRC`, and with nothing installed into py311.
     `test_pinned_runtime_versions` must now pass.

   If either fails, follow R4.

The maintenance is done only when all of these hold, each recorded below:

- step 7 printed `OK apply` and `STATE after`;
- step 8 printed `OK services started`, which covers the Tide smoke, the
  MHW checks with the point query matching the baseline, and one Dask
  worker under its nanny with four clients;
- step 9 printed `OK no Dask fallback in the new lines of err.log`, and the
  py311 suite passed its pin test.

Anything less is not done, whatever the services appear to do. A problem the
owner accepts under R4 is recorded as an exception, not as a pass. This
maintenance changes only Zarr: it does not deploy the v0.3.2 runtime files.

What to record here:

- every `OK` line of steps 1 to 9, with times, and how long both APIs were
  down;
- `OK apply`, the freeze diff (`diff $D/freeze-before.txt $D/freeze-after.txt`
  shows only the zarr line), and `pip-check-before.txt` vs
  `pip-check-after.txt` (identical);
- `STATE after`;
- numcodecs 0.15.1 and numpy 2.2.4, unchanged;
- step 9's log check result and the py311 suite counts.

### Recovery

Status: none of these branches has run on VM24, and none is validated there.
The chains below ran only in sandboxes, with the scripts replaced by stubs
(`tests/test_deployment_doc.py`). `py311_zarr_align.sh rollback` ran only
against a stand-in Python in its tests; real pip only showed why it needs
`--force-reinstall`, and the isolated 2026-10-27 container run rolled back to
2.18.6 with pip. The PM2 rehearsal ran neither `apply` nor `rollback`.

Where each failure leaves things:

| Step fails | Services | Package | Next |
|---|---|---|---|
| 1 to 5, preflights and baseline | all up | unchanged | R0: stop and report |
| 6, stop and full root check | some or all stopped | unchanged | R0: start them again |
| 7, root check, `apply`, or state check | all stopped | read `STATE` | R1 |
| 8, starts and checks | some started | 2.18.7 | R2: owner decides |
| 9, log check or py311 suite | all up | 2.18.7 | R4: owner decides |

Two rules hold in every branch:

- Services start only against a known package state: `STATE before` or
  `STATE after` from `py311_zarr_state.sh`. With `STATE unknown`, nothing
  starts.
- An API is never left stopped without a report. Whenever a branch ends with
  services stopped, report at once to the owner and the MHW maintainer:
  which apps are stopped, the failed step and its output, and the `STATE`
  line.

**R0. Step 1, 2, 3, 4, 5, or 6 fails (nothing installed).** Before the stop,
nothing changed: report and stop. If step 6 failed part way, py311 is
unchanged, so bring back whatever was stopped:

```bash
"$ZSTATE" --dir "$D" --expect before \
  && start_services "$D-r0-smoke" "$D-mhw-r0"
```

If this also fails, stop and report.

If the failure was the copy's hash (`q.sh: FAILED`), or step 1 failed, the
checkout may have been altered. Run the step 1 check first: if it does not
print `OK checkout`, run nothing more from `$SRC`, including the block above,
and report. The owner decides how to bring the services back.

**R1. Step 7 fails.** Run `"$ZSTATE" --dir "$D"` (read-only) and act on its
`STATE` line:

- `STATE before`: the root check refused, `apply` refused before installing
  (it printed "nothing was installed"), or pip rolled its own install back
  and every zarr file matches 2.18.6 again. Run the R0 block, then report.
- `STATE after` or `STATE unknown`: stop and report. Do not start anything.
  The services stay stopped until the owner decides: roll back (R3), or,
  for `STATE after` only, accept 2.18.7 and run step 8.

**R2. Step 8 fails.** The package was validated, so the state is known.
Stop and report the failed check before doing anything else; the owner
decides between:

- rolling back (R3), recommended when a Tide or MHW check failed;
- starting the rest against 2.18.7 after the cause is understood, with
  `"$ZSTATE" --dir "$D" --expect after && start_services "$D-r2-smoke" "$D-mhw-r2"`.
  It starts only the apps still stopped or errored and repeats every check.

If a Dask app failed to start, MHW would fall back to local execution if it
were started without Dask; that needs the MHW maintainer's agreement and is
not a default.

**R3. Rollback is selected.** Make py311 quiescent again, roll back, require
the `before` state, and start everything. The MHW point query must still
match the pre-apply baseline:

```bash
stop_services \
  && "$ALIGN" rollback --dir "$D" \
  && "$ZSTATE" --dir "$D" --expect before \
  && start_services "$D-rollback-smoke" "$D-mhw-rollback"
```

`stop_services` skips apps that are already stopped and stops errored ones.
If it, `rollback`, or the state check fails, stop and report; nothing starts.
That is the end of what this run sheet covers.

**R4. Step 9 finds a problem.** The services are up on 2.18.7 and stay up:
restart nothing and do not roll back. Report the finding; the owner decides
whether to accept it, or to roll back with R3. If the Dask fallback message
appeared, tell the MHW maintainer too. If the log check failed without
finding the message, nothing is known about the fallback: the log could not
be read, or its new lines could not be identified. Read `err.log` by hand
and report what it shows; the check still did not pass.

**Another attempt.** It needs a new sign-off of the window.

- After R0 with no `apply`, `$D` stays valid. The `$D-mhw-before` and
  `$D-mhw-r0` directories already exist and the MHW check refuses an
  existing output directory, so rename them aside with `mv`. Preflight B's
  PID 96946 no longer holds if MHW was restarted: resolve the interpreter
  again (see "MHW facts") and update that PID.
- Once `apply` has started, `$D` has its `applied` marker and accepts no
  second `apply`, even after a failed install or a rollback. Run `prepare`
  into a new directory first (see "Procedure"), which the owner must
  approve, because it downloads the wheels again.

Follow-up after the window: the next `update_mhw.sh` run on the 18th will be
the first MHW append under 2.18.7, which the isolated round trip covered.
Check its log and cron mail.

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
  owner authorized aligning Zarr only; see "Production run sheet" for the
  steps and the items still to settle.
- MHW (for its maintainer, not part of the Zarr maintenance):
  - the monthly `update_mhw.sh` cron cannot find `pm2`, so its
    `pm2 restart mhwapi` has not run since at least July 2026;
  - the stored `PATH` of `mhwapi` now begins with entries from an operator
    tool session, two of which no longer exist;
  - the MHW listener on 8030 serves a certificate that expired on
    2023-08-25.
- The internal 8040 listener's certificate expired on 2023-08-25 (the public
  443 frontend is valid). Whether and how to replace it is an owner
  decision; it is not part of the v0.3.2 runtime manifest.
- `conf/https_smoke.sh` defaults to `--prod-port 8040`; consider changing
  the default to 443 so the safe choice is the default. For now, the
  runbook passes `--prod-port 443` explicitly.
- Parameterize host-specific paths in `conf/start_app.sh` (the launcher
  tests currently rewrite them in a sandbox copy).
- Remove `--reload` from the production launcher.
- Remove the `tide` entry's `pre_stop` from `conf/ecosystem.config.js`. PM2
  5.4.2 never runs it, so it is dead configuration whose broad `tide_app`
  kill pattern would matter only under a PM2 version that did. Until then,
  the Zarr run sheet's preflight B requires the daemon to be 5.4.2.
- Clean up merged branches and VM24 temporary directories once the above is
  settled.
