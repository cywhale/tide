# TPXO10 Migration Workspace

`dev_tpxo10/` contains the TPXO10-atlas-v2 conversion, validation, benchmark,
and handoff artifacts for the v0.3.0 Tide API migration. Runtime code lives
in the repo root (`src/`, `tide_app.py`); this directory is the conversion
and evidence workspace.

## Environment

Use the local `uv` environment in this directory for conversion/golden tools:

```bash
cd dev_tpxo10
uv sync
uv run pytest
```

This environment intentionally uses pyTMD 3.x. The production API runtime
uses the repo-root environment and remains on pyTMD 2.2.8; the Zarr store is
the version boundary.

## Important Files

- `scripts/convert_to_zarr.py`: deterministic TPXO10 NetCDF -> Zarr converter.
- `scripts/verify_against_netcdf.py`: T-B NetCDF/Zarr verification.
- `scripts/golden_check.py`: T-D1 source-reader golden checks.
- `scripts/benchmark_old_new_api.py`: old/new API performance benchmark.
- `scripts/tf_observation.py` and `scripts/fetch_tf_observations.py`: T-F observation gate.
- `scripts/select_tf_noaa_stations.py`: reproducible NOAA station-panel selection manifest.
- `TESTING.md`: full G0->G3 evidence log.
- `STAGE3_G3_CLOSEOUT.md`: pre-deployment close-out and Codex deployment handoff.
- `benchmarks/`: tracked JSON gate evidence.
- `manifests/`: source NetCDF SHA-256 provenance.
- `plot_tide.ipynb`: manual visual QA notebook for tidal-current maps.

## Data Policy

Large data are not committed:

- Source NetCDF files live under `data_src/`.
- Canonical runtime stores live under `data/` (`data/tpxo10.zarr`,
  `data/tpxo9.zarr`) and are gitignored.
- Temporary conversion stores under `dev_tpxo10/stores/` are gitignored.

Do not commit `.env`, credentials, tokens, raw API responses with secrets, or
temporary `.zarr.partial` directories.

## Current Status

As of v0.3.0 pre-deployment, all binding gates are green:

- G0 source/environment inspection.
- G1 conversion contract and chunk/open-mode decisions.
- G2 global store conversion and deterministic verification.
- G3 runtime migration, performance warm/cold, T-D2 prediction golden, and
  NOAA T-F observation gate.

Deployment to VM24 is not part of this workspace. Codex owns the rollout:
isolated worktree, out-of-band `data/tpxo10.zarr` transfer, service switch,
smoke tests, and rollback drill. See `STAGE3_G3_CLOSEOUT.md`.

## Visual QA Notebook

`plot_tide.ipynb` is a manual diagnostic notebook converted from selected
legacy plotting work. It compares tidal-current map patterns between stores
and highlights regions where TPXO10 and TPXO9 may differ. It is useful for
human inspection but is not a binding gate; binding correctness is covered by
the scripts and evidence listed above.
