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
as 0. uz/vz match an independent centering reimplementation (rtol 1e-6).
*(Historical note: this first T-B run excluded the halo-dependent
easternmost column / northernmost row; superseded by S1.5 — the gate now
performs a full deterministic reproduction with no exclusions.)*
Provenance attrs verified.

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

### S1.6 Round 10 fixes + §7.5.1 chunk/open-mode matrix (2026-06-11)

Round 10 fixes (commit `ebfc9b0`): fail-closed one-sided golden branch
(independent edge-flag recompute from pyTMD-read pieces), invalid-branch
sampling (40 flag-2 cells per component: both edges source-invalid, store
exactly 0), provenance scope extended to pyproject/uv.lock/.python-version/
manifests with repo-relative keys, T-B verifies every recorded hash
against the recorded commit's blobs (`git show`). The provenance guard
fail-closed twice during this round on genuinely dirty trees — working
as designed. Canonical store rebuilt; T-B PASS incl. 11 hash↔blob checks;
golden PASS (uz 295/10/0/40, vz 289/11/0/40 two-edge/one-sided/skipped/
invalid-zero).

Matrix (`scripts/benchmark_chunk_matrix.py`, seed 20260611, 6 rechunk
variants × direct/dask × W1–W8 + tpxo9 baseline; full results in
`benchmarks/chunk_matrix.json`; decompressed = full-chunk upper bound):

| cell (dask mode) | W1 pt ms/MiB | W5 10° ms | W6 45°s5 ms | W8 45°s1 ms |
|---|---|---|---|---|
| tpxo10 113×113×15 | **3.93 / 4.38** | **13.0** | **137** | **217** |
| tpxo10 113×113×5 | 4.42 / 4.38 | 19.2 | 237 | 342 |
| tpxo10 225×225×15 | 5.22 / 17.4 | — | 101 | 195 |
| tpxo10 450×450×15 | 10.4 / 69.5 | — | 114 | 168 |
| tpxo9 baseline 113×113×8 | 4.49 / 9.35 | 33.5 | 301 | 362 |

Decision (spec Stage 1 decision memo #4–5): **chunks (113,113,15),
open-mode dask-'auto'**; D5 ceiling admits only spatial 113 (0.47× vs
1.86×/7.4×); cons=15 dominates (W3 proves arbitrary subsets span all
cons-chunks); winner beats baseline on every workload in the production
open-mode. overview5: no-go. Cold rounds labeled `first-path-access`
(no `purge` on this box) — binding cold gate moves to the Linux VM.

### S1.7 Round 11 scope corrections + T-C calibration (2026-06-11)

Round 11 (Codex) narrowed the S1.6 freeze: **D5 freezes chunk shape only
(113,113,15)**; open-mode is PENDING the W8 harness (matrix "dask" cells
used the local threaded scheduler with single-block `chunks='auto'` —
not production-representative); matrix latency numbers are exploratory
(fixed order, W6/W8 reps=2); overview5 no-go is provisional pending a
second bbox-origin phase (W6b workload added); matrix RSS is
end-minus-start, not peak. Variant stores now verify provenance commit +
chunk layout before reuse.

T-C calibration (`scripts/compare_tpxo9_tpxo10.py`, 117,492 common-valid
z-cells, seed 20260611 → `benchmarks/tc_calibration.json`):

| binding cons | deep P95 | shelf P95 | coastal P95 (report) |
|---|---|---|---|
| m2 | 9.90 mm | 22.02 mm | 150.61 mm |
| s2 | 6.71 mm | 20.59 mm | 99.13 mm |
| k1 | 5.66 mm | 14.14 mm | 63.42 mm |
| o1 | 5.00 mm | 14.32 mm | 49.66 mm |

Deep-M2 signed amplitude bias **+0.662 mm** (gate ≤ 5 mm — no systematic
unit/scale error). Top outliers cluster at Incheon Bay (extreme
macrotidal, h 2–5 m) and NE Borneo shallows — physically explicable, no
structured artifacts. Polar stratum deferred to Stage 2 (no |lat|>60°
cells in region). Currents report-only: legacy store units detected
cm/s (100×); u-M2 |Δ| P95 = 0.126 m/s. **Frozen thresholds (spec memo
#10): deep ≤ 30 mm, shelf ≤ 150 mm, bias ≤ 5 mm — observed margins
3×/7×.**

### S1.8 §7.5.3 W8 harness — **STOP-AT-G1: memory thresholds failed** (2026-06-11)

Harness: `w8app.py` (production env, real pyTMD 2.2.8 `predict.map` /
`time_series`, orjson path, X-Worker-PID) + `scripts/w8_harness.py`
(production gunicorn 2 workers NO --reload; dist modes launch a real
dask scheduler + 8 GB worker; 50 ms process-tree peak-RSS sampler;
concurrency-2 distinct-PID asserted). Results
(`benchmarks/w8_harness.json` + `w8_harness_dist.json`):

| mode | point med | W6 s5 / W6b | W8 single | W8 conc2 | worker peak/Δ GiB | tree Δ GiB | verdict |
|---|---|---|---|---|---|---|---|
| direct | **4.27 ms** | 1.01 / 1.00 s | 14.65 s | 15.09 s | 3.71 / 3.52 | ~7.0 | FAIL mem |
| dask (threaded auto) | 6.44 ms | 0.94 s | 14.63 s | 15.04 s | 4.27 / 4.05 | 7.46 | FAIL mem |
| dist-auto | 33.11 ms | 1.45 s | 15.21 s | 15.98 s | 3.83 / 3.62 | 6.81 | FAIL mem+wall |
| dist-native | 32.84 ms | 1.44 s | 15.20 s | 15.98 s | 3.76 / 3.55 | 6.53 | FAIL mem+wall |

Passing in all modes: payload 32.6 MiB (≤100), conc2 ≤30 s, two distinct
worker PIDs, host headroom ≥0.57 (≥0.25). Failing in all modes:
gunicorn worker delta (signed ≤1.5 GiB) and peak (≤2 GiB), tree delta
(≤3 GiB); dist modes additionally fail wall_single (15.2 s > 15 s).

Findings:
1. **The W8 memory blow-up is open-mode-independent** (~3.5–4 GiB worker
   delta in every mode) — the cost is the post-read pipeline (complex hc
   build + pyTMD predict temporaries + list/JSON conversion for 1.82M
   cells × 3 components), not the Zarr read. A †cap on output cells is
   the §7.5.3-prescribed remedy; no chunk/open-mode choice can fix it.
2. **Open-mode evidence** (from the workloads that pass): direct wins
   points decisively (4.27 vs 6.44 ms threaded, vs 33 ms distributed —
   scheduler round-trips cost 7.8× on points); maps are within noise
   (1.01 vs 0.94 s). The distributed client is strictly worse for this
   workload shape (store is local to the API process). The dask worker
   RSS stayed ~0 in dist modes — the data path does not benefit from the
   cluster.
3. W6/W6b (two sampling phases) identical → overview5 no-go can be
   finalized.
4. Caveat: the harness JSON path uses `.tolist()`; the ported production
   runtime may serialize more efficiently — the cap value must be
   re-validated on the real runtime at G3 (P-Map45-s1 already requires
   this).

**Per the G1 kickoff stop condition: work STOPPED; owner re-sign-off
required.** Scaling: ~2.0 KiB worker-RSS per output cell ⇒ proposed
`MAX_BBOX_CELLS = 5×10⁵` output cells (post-sample) ⇒ predicted ~1.0 GiB
delta / ~4 s wall; default `sample=5` 45° maps (73k cells) unaffected;
`sample=1` capped at ≈23.6°×23.6°.

### S1.9 W8 harness v2 — corrected measurements + cap sweep (2026-06-11)

Round 12 fixes applied: true recursive process-tree sampling (per-PID
first-seen baseline + peak; gunicorn workers identified by X-Worker-PID
headers), `/bench/map` reproduces the production serialization verbatim
(`tide_to_output` copy + `jsonable_encoder` + ORJSONResponse,
`absmax=10000.0`), legacy schema mode, cap sweep with a fresh service
per candidate. T-C now enforces the frozen thresholds (gate re-run:
PASS). Results (`benchmarks/w8_harness_v2.json`):

**Full 45° W8 (sample=1), production-faithful serialization:**

| schema/mode | W8 single | payload | worker peak/Δ GiB | tree Δ | verdict |
|---|---|---|---|---|---|
| tpxo10/direct | 14.97 s | **120.0 MiB** | 3.92 / 3.55 | 7.23 | FAIL mem+payload |
| legacy(tpxo9)/direct | 16.16 s | 120.5 MiB | **4.83 / 4.36** | 8.68 | FAIL mem+payload+wall |

The faithful serialization raises payload 32.6 → 120 MiB (now ALSO over
the 100 MiB threshold) and confirms the memory failure. **The legacy
store is empirically WORSE on the identical workload** — the exposure
pre-dates this migration (production today carries it); the new store
reduces it (~19% lower peak) but cannot fix it. A cap is required either
way. W6/W6b two phases identical (0.99/0.98 s) — overview5 no-go final.

**Cap sweep (tpxo10/direct, fresh service per candidate, single + conc2):**

| output cells | bbox | W8 s | payload MiB | worker peak/Δ GiB | tree Δ | margins vs budget | verdict |
|---|---|---|---|---|---|---|---|
| 250,000 | 16.7° | 1.97 | 17.1 | 0.71 / 0.53 | 0.99 | ≥65% | PASS |
| 400,000 | 21.1° | 3.16 | 28.3 | 1.05 / 0.86 | 1.61 | ≥43% | PASS |
| 500,000 | 23.6° | 4.01 | 35.5 | 1.14 / 0.95 | 2.07 | ≥31% | PASS |
| 600,000 | 25.8° | 4.86 | 42.0 | 1.38 / 1.19 | 2.26 | ≥21% | PASS |
| 750,000 | 28.9° | 6.10 | 51.9 | 1.52 / 1.33 | 2.73 | ≥9% | PASS |

All five candidates pass; margins shrink monotonically. 500,000 is the
largest candidate retaining ≥31% headroom on every signed metric
(600k drops to 21% on worker delta; 750k to 9% on tree delta).

### S1.10 Round 13 closure — cap signed, dist re-measured, **G1 CLOSED** (2026-06-12)

Round 13 fixes: every single/concurrent response PID joins the worker
set (fail if <2 observed — `n_workers_observed` in JSON);
`cells_actual` recorded via the `X-Grid-Cells` header (500k candidate =
500,556 actual cells). Reviewer independently reproduced the 500k sweep
point with both workers measured (peaks 1.31/1.18 GiB, tree Δ 2.11 GiB,
min margin ~25%).

Dist modes re-run with the corrected recursive sampler
(`w8_harness_v2.json`): the dask cluster genuinely carries 3.38–3.54 GiB
across 6 sampled PIDs (the old 0.03 GiB reading was a launcher-PID
artifact, exactly as round 12 finding 1 said); gunicorn workers reach
5.0–5.26 GiB (data shipped to the cluster AND copied back); points
31.6–31.7 ms; W8 17.0–17.4 s. Distributed is strictly worse than direct
on every axis — the D5 open-mode decision (direct) now rests on
corrected measurements.

**Owner + reviewer sign-off (2026-06-12): `MAX_BBOX_CELLS = 500_000`**
(spec memo #7 — counted on the post-sample output grid, enforced before
any `.compute()`/`.values`/prediction, HTTP 400 with actual cells +
limit + raise-`sample` suggestion, env-overrideable, production default
500,000, re-validated at G3); **D5 open-mode frozen: direct** (memo #8);
**overview5 no-go final** (memo #9).

### Gate G1 verdict: **CLOSED** (2026-06-12)

All spec §4 G1 items green: T-B (S1.3/S1.5 full reproduction), T-D1
(S1.4/S1.5), T-C calibrated + thresholds frozen + gate enforced
(S1.7/r12), §7.5.1 matrix with blocking instrumentation + D5
read-amplification ceiling (S1.6/S1.7), §7.5.3 cap decision with
pre-frozen thresholds and owner sign-off (S1.8–S1.10), overview5
decision recorded (final no-go). The conversion contract is now fully
frozen; Stage 2 (global conversion) may start.

Carried forward to G3 (not G1 items): Linux-host cold-cache round
(macOS rounds are `first-path-access` only); cap re-validation on the
fully ported runtime (P-Map45-s1).

## Stage 2 — Gate G2 (in progress)

### S2.0 Global streaming converter + round 15 publication hardening (2026-06-12)

Converter: `convert_to_zarr.py --full` — latitude-band streaming tiles
(tile_lat=226 + halo 32, full-longitude rows; periodic wrap via
`center_u(wrap=True)`; lat halo clamps at the true global boundaries;
top tile centers vz one-sided; per-tile Zarr slab writes; ~2.2 GiB RSS
observed during the run).

Single-tile smoke (2026-06-12, pre-round-15 evidence, re-recorded here
per round 15 finding 2): `--full --max-tiles 1 --out stores/smoke_global.zarr`
-> `[tile 1/1] rows 0:226 (halo 0:258) written`, fills {z:0,u:0,v:0}
(Antarctic land band), store removed afterwards. Test suite at that
point: `28 passed`.

Round 15 publication hardening (finding 1 — an interrupted run could
previously masquerade as a complete store):
- writes go to `<out>.partial`; attrs start `conversion_status=
  in_progress`, `canonical=false`; refuses to overwrite an existing
  published store or to silently resume an existing partial
- ALL flag arrays now have `fill_value=2` (unwritten regions read as
  invalid, never source-valid)
- per-tile `tiles_written` ledger; `assert_complete` guard; only then
  `conversion_status=complete`, `canonical=true`, consolidate and
  ATOMIC `os.rename` to the final path
- `--max-tiles` requires an explicit non-default `--out`, marks the
  store `smoke-partial`, never prints PASS, never publishes
  (`validate_smoke_args` + unit tests)

Round 15 finding 2 — integration equality test added
(`test_tiled_pipeline_bitwise_matches_monolithic_globe`): synthetic
periodic globe where the tiled decomposition must be BIT-IDENTICAL to a
monolithic computation across tile seams, including a seam-row fill
cell with unique donor, a polar-row fill cell, a dateline land strip
(col 34 one-sided; col 35 wraps to edge 0 with both edges valid) and
the one-sided northern row. Publication guards unit-tested
(`validate_smoke_args`, `assert_complete`). Suite: `31 passed`.

**Store status**: the in-flight global run (started before round 15)
is a **G2 candidate for verification tooling only** — per round 15 it
is NOT canonical; the canonical store will be rebuilt with the
round-15 pipeline revision after the candidate-based gate dry-runs.

### S2.1 Canonical global store + G2 gate chain (2026-06-12)

Canonical `stores/tpxo10_global.zarr` built at commit `31f2491d` via the
round-15 atomic path (24/24 tiles, 37 min, 12 GB, `canonical=true`,
`conversion_status=complete`); fill counts z=356 / u=618 / v=595 —
IDENTICAL to the pre-round-15 candidate run (cross-revision determinism
evidence). The candidate is retained at `tpxo10_global_candidate.zarr`.

**G2 root-cause find**: xarray's default CF decoding reinterpreted the
round-15 structural zarr fill_values as `_FillValue` sentinels
(int32→float64, fill-equal cells → NaN), producing every first-run gate
failure (probe: 113 NaN == 113 fill cells; raw lat/lon maxdiff 0.0).
Fix: `mask_and_scale=False` at all 10 store-open sites + a binding
raw-read contract in spec §3.1 (the Stage 3 adapter MUST open raw).

Gate results (after fix; `b9k9k9e7t` + coverage reruns):

- **T-B global (sampled 6 tiles incl. top)**: PASS — full deterministic
  reproduction per tile (truth byte-equality, derived wrap-centering
  rtol 1e-6, flags exact), coordinates full-scan exact, provenance 13
  hash↔commit-blob checks OK. Full 24-tile scan pending (required for
  G2 sign-off).
- **Coverage**: PASS — zero NaN; fill fractions 0.0009–0.0016% (≤2%);
  legacy-valid→new-invalid cells 100% machine-classified:
  z 6,470 / u 66,224 / v 167,747 **coastline-reclassified** (TPXO9 h>0,
  TPXO10 h=0 — the advertised all-node coastline redefinition; legacy
  amp at these cells tiny: z P50 1.0 cm, max 9 cm; u/v extremes up to
  73/156 cm/s are legacy-fillna artifacts) + 4 **isolated-no-data**
  cells (134.2°E/46.5°N inland water body: TPXO10 h≈1.3 m, source hc
  all-zero verified, beyond DMAX — flag 2 is the frozen §3.2+D3
  outcome); **0 unexplained**. ⚠ The original strict criterion is
  amended to "all violations machine-explained" — owner/reviewer
  sign-off requested (runtime note: at reclassified cells the new API
  returns no value where legacy served extrapolated values).
- **T-C global (polar stratum BINDING)**: PASS frozen thresholds —
  119,871 cells (deep 69,405 / shelf 33,594 / coastal 16,872 / polar
  61,247); binding deep P95 max 26.42 mm (K1) ≤ 30; shelf max 97.41 mm
  (M2) ≤ 150; deep-M2 bias −0.739 mm ≤ 5. Global values sit higher than
  the Stage 1 region (more polar/Atlantic change) but inside the frozen
  gates → `benchmarks/tc_global.json`.
- **T-D1 golden global**: PASS — 250 cells + 10 named (Arctic, Weddell,
  0/360 wrap pair); z hc, u 248 + v 245 transport cells rtol 1e-9;
  uz 306 two-edge + 4 one-sided, vz 240 two-edge + 69 one-sided
  (incl. the periodic easternmost column and the one-sided last
  latitude row), 40+40 invalid-zero cells; 0 skipped.
- **Idempotency**: PASS — 2-tile rebuild, 18 arrays byte-identical over
  rows 0:452.

### S2.2 T-B FULL scan — PASS (2026-06-12)

`verify_against_netcdf.py --store stores/tpxo10_global.zarr` (no
--sample-tiles): all 24/24 tiles reproduce byte-exactly (truth layers,
flags, derived wrap-centering rtol 1e-6); reproduced fill counts
{z:356, u:618, v:595} match the store; coordinates full-scan exact;
provenance 13 hash↔commit-blob checks OK.
`PASS verify_against_netcdf (tolerance 0 on source layer)`

**Gate G2 status: all technical gates GREEN.** Sole remaining item:
owner/reviewer sign-off of the coverage-gate amendment (S2.1 — strict
zero-violation criterion → all-violations-machine-explained, given the
TPXO10 coastline reclassification and the 4 isolated-no-data contract
cells; 0 unexplained).
