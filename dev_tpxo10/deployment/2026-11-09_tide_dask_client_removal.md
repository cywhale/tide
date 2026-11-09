# Tide Dask-client removal deployment, 2026-11-09

The owner authorized deployment after PR #12 was merged. This record covers
the rollout of PR #11's merge commit to VM24. It does not assign a release
version or create a tag. The API version remains v1.1.0.

## Scope

- Target commit: `f9c1e47f14f3048a5a04d117c4cb8c05943bc5f8`.
- Release directory: `/home/odbadmin/python/tide_tpxo10_atlas_v2`.
- Only Tide's three-file runtime manifest changed. MHW, the shared Dask
  scheduler and worker, TLS credentials, py311 packages, stores, and data
  were not changed.
- No root or sudo operation was needed. No restore or recovery ran.
- The deployment did not remove the old `tide-client` certificate/key files;
  credential retirement is a separate host-level decision.

All times below are VM24 local time, Asia/Taipei (UTC+08:00).

## Pre-deployment validation

At 16:47, after the host security updates, the read-only baseline found:

- PM2 v5.4.2 daemon PID 3352 alive; `tide`, `mhwapi`, `dask-scheduler`, and
  `dask-worker` online.
- PIDs: Tide 716011, MHW 611920, scheduler 609785, worker 610501.
- Port 8041 was free. The public 443 point smoke returned HTTP 200.
- The canary certificate SAN was `eco.odb.ntu.edu.tw`, expiry was
  2026-12-10, and the derived public-key hashes of certificate and key
  matched. The private-key contents were not printed or recorded.
- The release runtime hashes were recorded before any production write.

A fresh detached checkout was created at
`/home/odbadmin/tmp/tide-dask-remove-f9c1e47f14f3048a5a04d117c4cb8c05943bc5f8`.
Its HEAD was the target commit and its Git status was clean; the existing
TPXO10 and TPXO9 stores were linked for tests, not copied or modified. The
transferred Git bundle SHA-256 was
`ef895d27d96a0101768974631e81609ddf6f9982143cfd98ef21dfe2e812727f`.

### VM24 py311 suite

Command, with Dask-related environment variables unset and without
`TIDE_DASK_DISABLE`:

```bash
/home/odbadmin/.pyenv/versions/py311/bin/python -m pytest -rs
```

Result: **564 passed, 0 skipped, 2 warnings**, exit 0, in 131.03 seconds.
The warnings were the existing NumPy ABI-size and Polars CPU-feature
warnings. No packages were installed. Runtime versions included Python
3.11.4, Zarr 2.18.7, NumPy 2.2.4, numcodecs 0.15.1, and PyTMD 2.2.8.
The pytest log is retained on VM24; its SHA-256 is in the companion
`.sha256` file.

### Canary and scratch rehearsal

The non-PM2 canary ran from the target checkout on `127.0.0.1:8041`, with
the existing TPXO10 store and verified HTTPS certificate. All four Dask
TLS/address variables and `TIDE_DASK_DISABLE` were unset in the canary
process. It returned HTTP 200 and a 1101-byte response identical to
production 443 (same SHA-256). The scheduler's `Client-tideapi-*` count was
2 before and during the canary, so the canary added no Tide Dask client.
The canary had no Dask connection-failure warning, was stopped by its own
PID 1043117, and port 8041 was confirmed free afterward. The temporary
mTLS observer was disconnected after each scheduler-list read and submitted
no task.

One initial observer query failed because its callback parameter did not
match the keyword Dask supplied. The observer's `finally` block closed that
client; the corrected read-only query then passed. The failed query submitted
no task and did not change service state.

The current deployment helper was rehearsed against a scratch release
under `/home/odbadmin/tmp`:

- Deploy replaced exactly the three manifest files and verified them
  against the target commit.
- Reusing the successful `WORK` was refused without changing the scratch
  release.
- Restore recovered the scratch release's original hashes and modes.
- A tampered source at the target commit was rejected before any scratch
  file was replaced.

The scratch release, work directories, logs, and tampered-source test copy
are retained on VM24; none is the production release.

## Production deployment

The deploy, restart, and HTTPS smoke ran as one checked chain from
17:00:10–17:00:21. `WORK` is
`/home/odbadmin/tmp/tide-dask-remove-deploy-20261109T170010`; keep it because
its three backups are the rollback source for this attempt.

| Runtime file | Before | Deployed target commit |
|---|---|---|
| `conf/start_app.sh` | `c96af310331b44ea519de383007bd9cde4d3d948f5da5ddce8f536659d9d8c03` | `be4b2d2ee8b08f03e786610d9183106c97fe8acec6d1131ef2774ccbd989c441` |
| `tide_app.py` | `6a508d547f18ee76bb2bcab5462a4cc4543ab1c277283ec91df1d47fcec8d4a6` | `396c11c32083711e349e7a4ecdba876970faa6c028e0380557ebe6ab60c6fac5` |
| `src/dask_client_manager.py` | `a8fc7a05e8042ca69c0b8c9ad4224867c05f397fa978550a2b9efe64c43a5a46` | `553c0ac8d94ed7d8a78905f27a1b352cd82600ad3a6f9aeaa5fcf3fb3fbcbbf5` |

The helper printed `OK deploy`. The Tide restart changed PID 716011 to
1044542 and restart count 3 to 4; it remained stable for 10 seconds and
`OK restart` was printed. The public HTTPS smoke returned HTTP 200
(`PASS prod`), 1101 bytes. The response SHA-256 matched the canary and
pre-deployment production response.

## Post-deployment checks

At 17:01:

- All three manifest hashes matched the target commit. Every non-manifest
  runtime file matched the pre-deployment baseline.
- Tide was online as PID 1044542, restart count 4. The Tide process had no
  Dask scheduler/address or TLS variables in its environment.
- MHW remained online as PID 611920 (restart count 1); scheduler remained
  PID 609785 (count 2); worker remained PID 610501 (count 0).
- A fresh scheduler client-list read showed **zero** `Client-tideapi-*`
  entries. The two MHW clients remained. The observer disconnected after
  the read.
- The shared py311 `pip freeze` was 232 lines both before and after, with
  identical SHA-256; no package operation ran.
- The active Tide error log retained the same inode. Only bytes after the
  pre-deployment offset were checked; no Dask connection-failure or startup
  failure marker was found. The check did not inspect rotated logs.
- Port 8041 was free; the target checkout remained clean.

No new release version or tag was assigned. The v0.3.2 deployment `WORK`
was not reused. The new production `WORK` and its backups, the validation
checkout and bundle, pytest/canary/smoke logs, scratch rehearsal artifacts,
and freeze snapshots remain on VM24. No cleanup or credential retirement
was performed.

The companion `.sha256` file records the retained logs, responses, freeze
snapshots, bundle, and deployment backup record. No private key, certificate
content, Dask key material, or service environment dump is included.
