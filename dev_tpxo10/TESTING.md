# dev_tpxo10 TESTING log (spec v0.3.0)

Reproducible verification evidence, one section per stage gate
(convention copied from `gebco/dev2026/TESTING.md`).

## Stage 0 — Gate G0 (2026-06-11)

### Environments

| Env | Path | Python | Key pins |
|-----|------|--------|----------|
| production | repo root `pyproject.toml` | 3.11 | pyTMD v2.2.8, zarr 2.18.7 (frozen from requirements.txt; D6/D7) |
| conversion | `dev_tpxo10/pyproject.toml` | 3.12 | pyTMD v3.0.6, zarr 2.18.7, numpy 2.4.6, xarray 2026.4.0 (D7) |

Both pinned by committed `uv.lock` + `.python-version` (`.gitignore`
exceptions added — the repo-wide `**/*.lock` / `**/.python*` rules would
otherwise drop them).

### G0.1 Production env: pytest green, app importable, API serves tpxo9

```
cd ~/proj/tide && uv sync && uv run pytest
```

Result: `3 passed` —
`test_app_importable`, `test_pinned_runtime_versions` (pyTMD v2.2.8 /
zarr 2.18.7), and `test_api_serves_from_tpxo9_zarr` (FastAPI TestClient,
real lifespan, GET `/api/tide` point query against `data/tpxo9.zarr`,
HTTP 200 with non-empty payload). Existing API behavior unchanged.

### G0.2 Conversion env: unit tests

```
cd ~/proj/tide/dev_tpxo10 && uv sync && uv run pytest
```

Result: `5 passed` — `expected_files` inventory builder; node-offset
checker passes on a conventional grid and detects: flipped (eastern-edge)
u convention, non-periodic longitude span, non-uniform latitude spacing
(the historical tpxo9 `lat[2700]` corruption shape).

### G0.3 Source inspection (spec §2 schema + D12 node offsets + manifest)

```
cd ~/proj/tide/dev_tpxo10 && uv run python scripts/inspect_source.py
```

Result:

```
[1/4] inventory: OK 31 files
[2/4] schema (31 files): OK
[3/4] D12 node offsets: OK (u=west edge, v=south edge, periodic lon)
[4/4] sha256 manifest: OK 31 files, 21705323720 bytes -> dev_tpxo10/manifests/tpxo10_atlas_v2.sha256.json
PASS inspect_source
```

Node-offset directions (`lon_u = lon_z − 1/60°`, `lat_v = lat_z − 1/60°`)
now confirmed three ways: manual NetCDF read (spec §2), independent
reviewer verification (review round 5), and this scripted assertion.

The SHA-256 manifest (`manifests/tpxo10_atlas_v2.sha256.json`, committed)
is the provenance input for the D8 store attrs.

### G0 verdict: PASS (all three gate conditions)

### Round 8 review fixes (2026-06-11) — G0 CLOSED

Codex Stage 0 code review independently re-verified all G0 evidence
(both pytest suites, live tpxo9 smoke, full SHA-256 manifest
recomputation matching the committed file, all 30 `con` identities) and
required three fixes before closing G0:

1. All inspector coordinate/spacing/offset comparisons now use
   `rtol=0.0` (default NumPy rtol admitted ~3.6e-3 deg near lon 360°);
   high-longitude regression test added.
2. Semantic identity checks added: in-file `con` must match the filename
   constituent; every `h_*/u_*` coordinate array must be bit-exact equal
   to the grid file's.
3. Spec §7.5.3: Gunicorn 2 GiB peak re-labeled a project safety budget
   (PM2 supervises the shell script, not Gunicorn workers); W8 must run
   without `--reload`; Dask restated as 2 GiB delta / 4 GiB peak.

Re-run results:

```
cd ~/proj/tide/dev_tpxo10 && uv run pytest            # 7 passed
uv run python scripts/inspect_source.py --no-hash
[1/4] inventory: OK 31 files
[2/4] schema + identity (31 files: con matches filename, coords bit-equal grid): OK
[3/4] D12 node offsets: OK (u=west edge, v=south edge, periodic lon)
[4/4] sha256 manifest: SKIPPED (--no-hash)   # manifest unchanged; round-8 reviewer recomputed and matched
PASS inspect_source
```

## Stage 1 — Gate G1 (pending)

Blocked on G1 kickoff confirmation per spec status note (round-6
amendments: §3.0 quantization rule, §7.1 no-clamp default + edge-depth
survey) and §7.5.3 threshold freeze, before the converter writes data.
