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

## Stage 1 — Gate G1 (in progress)

G1 kickoff confirmed by owner 2026-06-11 (recorded in spec §4 Stage 1
block) with two STOP conditions: clamp needed, or any W8 threshold fails.

### S1.1 Edge-depth / velocity-outlier survey — no-clamp PASS

```
cd ~/proj/tide/dev_tpxo10 && uv run python scripts/survey_edge_depth.py
```

Region 104–151°E / −1–46°N, §3.2-valid nodes: u-nodes n=1,402,333,
min positive hu = 1.57 m (P0.1 = 5.0 m, zero cells < 1 m); v-nodes
n=1,397,975, min hv = 1.50 m. Max per-constituent speed amplitude
|1e-4·U/h| = 1.291 m/s (u, M2), 1.174 m/s (v, M2); P99.99 = 0.602 m/s.
Top outliers all in known strong-current channels (Surigao Strait,
Sulu archipelago, San Bernardino). `PASS no-clamp policy (<= 5 m/s)` —
**STOP condition NOT triggered; unclamped `U/h` division stands.**

### S1.2 Prototype conversion (default chunks 113×113×15)

```
uv run python scripts/convert_to_zarr.py   # region 104,151,-1,46
```

Output `stores/tpxo10_proto.zarr`, 474 MB, interior 1409×1410, halo 32.
Fill counts (interior): **z = 0** (TPXO10's all-node coastline definition
holds — no z holes in this region), u = 77, v = 56 (vs ~204,969 problem
points in the TPXO9-era conversion). uz flags: 1,397,363 native /
11,122 derived / 578,205 invalid.

### S1.3 T-B NetCDF↔Zarr full regional scan — PASS

```
uv run python scripts/verify_against_netcdf.py
```

Coordinates + hz/hu/hv exact; flags equal deterministic recompute;
flag-0 truth layers bit-exact (tolerance 0): z 1,408,771 / u 1,402,333 /
v 1,397,975 cells; flag-1 cells confirmed source-invalid; flag-2 stored
as 0. uz/vz match an independent centering reimplementation (rtol 1e-6;
easternmost column / northernmost row excluded — halo-dependent, covered
by golden checks). Provenance attrs verified.

### S1.4 T-D1 golden checks vs pyTMD 3.0.6 readers — PASS

```
uv run python scripts/golden_check.py   # seed 20260611
```

246 flag-0 z-cells (deep/shelf/coastal strata + 6 named locations incl.
Taiwan Strait, Surigao). z hc rtol 1e-9 OK; u transport rtol 1e-9 OK
(note: pyTMD 3.0.6 `open_atlas_dataset(group='u')` returns *transport*
in cm²/s at this layer — depth division happens in higher-level
accessors); hu/hv equivalence OK (pyTMD masks land to NaN, store keeps
raw 0 — finite cells exact); D12 centering OK for 235 two-edge and 60
one-sided coastal cells (rtol 1e-5). Pipeline unit tests: 18 (toy-grid
quantization ties/overflow-abort/byte-identity, validity, fill band,
inpaint determinism, no-clamp edge velocity, centering incl. periodic
wrap + inpainted-edge precedence + halo≡global toy proof).

### S1.5 Round 9 review fixes + canonical prototype rebuild (2026-06-11)

Codex Stage 1 core review (independently re-ran all gates, re-converted
with 25/25 arrays identical, recomputed both halo boundaries) required
four fixes, all applied at commit `c359983`:

1. Provenance fail-closed: converter aborts on dirty/untracked pipeline
   sources and records per-file SHA-256 of all 6 pipeline source files.
2. T-B is now a FULL deterministic reproduction (source halo + recomputed
   flags + recomputed inpaint, byte equality across every flag class) and
   the derived-layer scan covers the easternmost uz column / northernmost
   vz row — no exclusions.
3. T-D1 covers v transport + vz centering; centering references built
   exclusively from pyTMD-read transport and edge depths; boundary cells
   included via global reads. Spec corrected: `open_atlas_dataset`
   (group='u'/'v') returns transport (cm²/s) at this layer.
4. T-A additions: transpose round-trip, provenance fixtures, two-run
   byte-identity (1° region, fixed timestamp) — `27 passed` total; the
   byte-identity test also failed-closed correctly while round-9 edits
   were uncommitted, passing only on the clean tree.

Canonical prototype rebuilt at `c359983` (attr `pipeline_git_commit` =
`c3599839...` verified). Re-run results:

```
verify_against_netcdf.py : PASS (full scan, truth layers byte-equal full
  reproduction; uz/vz full scan incl. boundary column/row)
golden_check.py          : PASS (z hc; u 239 + v 238 transport cells;
  uz 295 two-edge + 10 one-sided; vz 290 two-edge + 10 one-sided;
  0 skipped-inpaint; boundary cells included)
```

### S1 remaining for G1

- §7.5.1 chunk/open-mode matrix (blocking instrumentation) → D5 freeze
- §7.5.3 W8 cap decision (production-like Gunicorn, no --reload,
  concurrency-2 PID-verified) against the signed absolute thresholds
- T-C tpxo9↔tpxo10 cross-version calibration → threshold freeze
- decision memos into the spec (mask rule, inpaint params incl. observed
  fill counts, chunk winner, overview5 no-go default)
