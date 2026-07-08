# Legacy TPXO9 conversion scripts (archived v0.3.0)

These are the original TPXO9-atlas-v5 → Zarr conversion + missing-data
fill scripts, retired by the v0.3.0 migration to a single deterministic
streaming converter under `dev_tpxo10/`. They are kept for historical
reference only and **must not be run** — the v0.3.0 pipeline
(`dev_tpxo10/scripts/convert_to_zarr.py`) supersedes them and the runtime
no longer depends on the fillna step they implemented.

- `extract_parallel.py` — old per-chunk `ATLAS.extract_constants` extractor
- `zarr_fillna_parallel.py` / `zarr_fillna_concurrent_lock.py` /
  `zarr_fillna_savefile.py` — post-hoc NaN re-extraction/fill
- `test_fillna_parallel_write.py` — test for the (now-deleted) fillna write path

- `simu_result01.py` — TPXO9-schema QC script (old get_tide_map signature; rewrite against the adapter if revived)
