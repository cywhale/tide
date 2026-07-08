# v0.3.0 Stage 3 / Gate G3 — Close-out & Deployment Handoff

Post-deployment note: VM24 production cutover was completed after this
pre-deployment close-out. See `dev_tpxo10/DEPLOYMENT_VM24.md` for the
current runtime layout and rollback procedure.

Branch: `feature/v0.3.0-stage3-adapter` (merge-base with `origin/main`:
`bc9e9d4`). Spec: [`specs/v0.3.0_tpxo10_migration_plan.md`](../specs/v0.3.0_tpxo10_migration_plan.md).
Full per-gate evidence: [`TESTING.md`](TESTING.md).

This branch completes the v0.3.0 TPXO9-atlas-v5 → TPXO10-atlas-v2
data-source migration through Gate G3. **All binding gates G0→G3 are
green.** Deployment rollout (VM24 worktree, store transfer, service
switch, rollback verification) is handed to Codex — see the checklist at
the end. This document does NOT touch the VM24 production process or repo.

---

## 1. Gate G3 close-out summary

| Gate | Result | Evidence |
|---|---|---|
| **G0** environments + source inspection | PASS | TESTING G0; `manifests/tpxo10_atlas_v2.sha256.json` |
| **G1** conversion contract frozen (chunks (113,113,15) int32, no-clamp, inpaint, `MAX_BBOX_CELLS=500_000`, D5 open-mode direct) | CLOSED | TESTING S1.x; `benchmarks/{chunk_matrix,tc_calibration,w8_harness*}.json` |
| **G2** global store built + verified | CLOSED | TESTING S2.x; canonical `data/tpxo10.zarr` @ `3c5ba21` |
| **G3 T-D2** golden prediction (pyTMD 2.2.8 ↔ 3.0.6 engine) | PASS, worst RMSE **0.001 mm** | `benchmarks/td2_result.json` |
| **G3 §7.5.2** warm core + P-Strip + cap revalidation | PASS (tpxo10 faster) | `benchmarks/perf_old_new.json` |
| **G3 §7.5.2** P4 cold-cache (Linux VM24, drop_caches) | PASS (tpxo10 faster) | `benchmarks/cold_cache_vm24.json` |
| **G3 T-F** observation, NOAA primary (34-station panel) | PASS (82.4% within 1cm, mean RMSE new ≤ old) | `benchmarks/tf_observation_noaa_20260617.json` |
| G3 T-F CWA 24h smoke (pipeline/diagnostics, NOT a gate) | validated | `benchmarks/tf_observation_cwa_20260617.json` |

Key migration properties verified:
- **Lossless + deterministic** conversion (T-B byte-equality, idempotent
  rebuild, cross-revision identical fills z356/u618/v595).
- **Zero prediction drift** across the pyTMD version boundary (T-D2 0.001 mm).
- **tpxo10 faster than tpxo9** on every dominant pattern, warm AND cold.
- **Honest missing data**: TPXO10 coastline reclassification serves NULL
  (not fabricated 0) where the model has no ocean cell — verified at the
  API for point/const/forecast.
- **Rollback**: the same runtime serves both schemas via the D9 adapter
  (`TIDE_ZARR_PATH` switch); dual-schema API golden tests pass.

---

## 2. How to reproduce (test commands + results)

Production env (repo root, pyTMD 2.2.8):
```
TIDE_DASK_DISABLE=1 uv run pytest            # 80 passed
```
Conversion env (dev_tpxo10, pyTMD 3.0.6):
```
cd dev_tpxo10 && uv run pytest               # 31 passed
```
Gate scripts (production env unless noted):
```
# T-D2 golden (ours producer, then pyTMD-3.x reference + gate)
TIDE_DASK_DISABLE=1 uv run python dev_tpxo10/scripts/td2_ours.py
cd dev_tpxo10 && uv run python scripts/td2_golden.py          # PASS 0.001mm
# §7.5.2 warm/strip/cap (needs data/tpxo9.zarr + data/tpxo10.zarr)
TIDE_DASK_DISABLE=1 uv run python dev_tpxo10/scripts/benchmark_old_new_api.py
# T-F NOAA primary (reproducible station selection -> fetch -> gate)
TIDE_DASK_DISABLE=1 uv run python dev_tpxo10/scripts/select_tf_noaa_stations.py
uv run python dev_tpxo10/scripts/fetch_tf_observations.py --source noaa \
  --stations <model-resolved ids> --begin 20260610 --end 20260614 \
  --out dev_tpxo10/benchmarks/tf_observations_noaa_<DATE>.json
TIDE_DASK_DISABLE=1 uv run python dev_tpxo10/scripts/tf_observation.py \
  --observations dev_tpxo10/benchmarks/tf_observations_noaa_<DATE>.json \
  --out dev_tpxo10/benchmarks/tf_observation_noaa_<DATE>.json   # gate_pass=True
```
Cold-cache (Linux host only; macOS cannot evict):
`sync; echo 3 > /proc/sys/vm/drop_caches` per read — see
`benchmarks/cold_cache_vm24.json`.

---

## 3. Tracked gate evidence (all committed, worktree clean)

`dev_tpxo10/benchmarks/`: `td2_result.json`, `perf_old_new.json`,
`cold_cache_vm24.json`, `tc_global.json`, `tf_observation_noaa_*.json`,
`tf_observations_noaa_*.json`, `tf_noaa_station_selection_*.json`,
`tf_observation_cwa_*.json`, `tf_observations_cwa_*.json`,
`cwa_station_meta.json`, `chunk_matrix.json`, `w8_harness*.json`.
Gitignored (regenerated/intermediate): `td2_ours.json`, `tf_result.json`.

---

## 4. PR description (for the merge to `main`)

> **v0.3.0 — TPXO9-atlas-v5 → TPXO10-atlas-v2 data-source migration (Stage 3 / G3)**
>
> Migrates the ODB Tide API store from TPXO9 (amp/ph) to TPXO10
> (native Arakawa C-grid Re/Im + D12-centered uz/vz), behind a
> schema-keyed store adapter (D9) so the runtime is schema-agnostic and a
> rollback is a single `TIDE_ZARR_PATH` env switch + restart. A unified
> bbox query planner enforces `MAX_BBOX_CELLS=500_000` before any
> materialization. Runtime stays on pyTMD 2.2.8; conversion uses pyTMD
> 3.0.6 (the Zarr store is the version boundary). Project tooling moved
> to `uv`; legacy TPXO9 conversion scripts archived to `dev/legacy_tpxo9/`.
>
> All gates G0→G3 green (see `dev_tpxo10/TESTING.md`): lossless+
> deterministic conversion, zero prediction drift (T-D2 0.001 mm),
> tpxo10 faster warm+cold, NOAA observation gate PASS, honest missing
> data (coastline reclassification → null, not 0). 80 production tests +
> 31 conversion tests pass.
>
> **Deployment** (VM24 store transfer + service switch + rollback) is a
> separate Codex-owned step — `data/tpxo10.zarr` is NOT in git (gitignored
> 12 GB store); see `dev_tpxo10/STAGE3_G3_CLOSEOUT.md` §5.

---

## 5. Deployment handoff checklist (for Codex — NOT executed here)

The migration code is merge-ready; deployment is operational and owned by
Codex. **Nothing below has been run against VM24 production by this branch.**

Pre-flight:
- [ ] Merge `feature/v0.3.0-stage3-adapter` to `main` (PR §4).
- [ ] The canonical store `data/tpxo10.zarr` (12 GB) is **gitignored** —
      transfer it out-of-band to the deploy host. Provenance: built at
      commit `3c5ba21` (ancestor of HEAD); attrs `tide_store_schema=
      tpxo10-cgrid-v1`, `canonical=true`, `conversion_status=complete`,
      24 tiles. Verify on the host with
      `verify_against_netcdf.py --store <path> --sample-tiles 6` (needs
      the source NetCDF) or at minimum check the attrs.
- [ ] Keep `data/tpxo9.zarr` in place on the host — it is the rollback store.

Binding runtime contracts the deployment MUST preserve (CLAUDE/spec):
- [ ] Store opened ONLY via the adapter (`store_adapter.open_store` /
      `get_zarr_path`) — never a hardcoded `open_zarr`. RAW READ
      (`mask_and_scale=False`) and DIRECT (`chunks=None`) are enforced by
      the adapter and must not be overridden.
- [ ] `TIDE_ZARR_PATH` selects the store; default resolves to
      `data/tpxo10.zarr`. Rollback = set `TIDE_ZARR_PATH=data/tpxo9.zarr`
      + restart (the adapter fail-closed-detects the legacy schema).
- [ ] `MAX_BBOX_CELLS=500_000` (env `TIDE_MAX_BBOX_CELLS`) — the planner
      rejects oversized maps with HTTP 400 before materialization.
- [ ] Dask: production keeps the shared scheduler; `TIDE_DASK_DISABLE=1`
      is a test/single-process convenience only.

Cutover:
- [ ] Deploy the merged runtime with `TIDE_ZARR_PATH=data/tpxo10.zarr`.
- [ ] Smoke the live endpoints: `/api/tide` point + bbox map (incl. a
      45° `sample=1` → expect HTTP 400 cap), `/api/tide/const` multipoint,
      `/api/tide/forecast`. Confirm an invalid/land cell returns null/empty
      (not 0).
- [ ] Rollback drill: switch `TIDE_ZARR_PATH` back to `data/tpxo9.zarr`,
      restart, confirm the same runtime serves the legacy store, then
      switch forward again.

Operator-facing behavior change to announce (change_log v0.3.0):
- [ ] At coastline-reclassified cells (TPXO10 land where TPXO9 served
      extrapolated values), the API now returns NO value instead of a
      fabricated 0 — a deliberate honesty improvement.

Follow-ups (not blockers): broader NOAA station panels / longer windows
(re-run the same harness); persistent VM34/VM24 deployment is rollout
work; an optional pyTMD-3.x runtime experiment behind the §7.5 gates.
