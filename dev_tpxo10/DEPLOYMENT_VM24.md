# VM24 TPXO10 Deployment Notes

Initial TPXO10 cutover: 2026-06-18. Formal v0.3.1 release-directory
cleanup: 2026-07-08. API version: v1.1.0. Project release: v0.3.1.

## Current Production Layout

- Host: VM24 (`odb24`, `192.168.2.24`).
- PM2 process: `tide`.
- Runtime cwd: `/home/odbadmin/python/tide_tpxo10_atlas_v2`.
- HTTPS bind: `127.0.0.1:8040`.
- Runtime store: `data/tpxo10.zarr` (TPXO10-atlas-v2).
- Legacy rollback repo/store: `/home/odbadmin/python/tide` with
  `data/tpxo9.zarr` (TPXO9-atlas-v5).

The old `~/python/tide` repository is intentionally kept unchanged as the
TPXO9 rollback anchor. Public users do not select TPXO9; it is deprecated for
public API use and retained only for operator rollback.

## Cutover Summary

The release was first staged and smoked on a separate local port. The
canonical TPXO10 store was copied into the release directory, then PM2 was
switched from the old cwd to the TPXO10 release cwd. Smoke checks passed:

- `/api/tide` point time series: 200.
- `/api/tide` small bbox map: 200.
- `/api/tide` 45-degree `sample=1`: 400 with `MAX_BBOX_CELLS`.
- `/api/tide/const` multipoint: 200.

Rollback drill passed: PM2 was switched back to `~/python/tide`, a TPXO9
point query returned 200, and PM2 was then switched forward to the TPXO10
release again.

The temporary release directory was later renamed to the formal runtime cwd
`~/python/tide_tpxo10_atlas_v2`; PM2 was restarted from that cwd and saved.

## Rollback

```bash
PM2=$HOME/.npm-global/bin/pm2
export PATH=/home/odbadmin/.pyenv/versions/py311/bin:$PATH
$PM2 delete tide
cd ~/python/tide
$PM2 start conf/ecosystem.config.js --update-env
$PM2 save
```

## Switch Forward

```bash
PM2=$HOME/.npm-global/bin/pm2
$PM2 delete tide
cd ~/python/tide_tpxo10_atlas_v2
$PM2 start conf/ecosystem.config.js --update-env
$PM2 save
```

## Operational Notes

- `conf/start_app.sh` resolves gunicorn in this order: `GUNICORN_BIN`, `uv run
  gunicorn`, PATH, then VM24 py311 fallback. This avoids relying on an
  interactive shell PATH.
- TPXO10 coastline reclassification can return no value where TPXO9 used to
  serve extrapolated values.
- Large bbox maps are capped by `MAX_BBOX_CELLS=500000` after applying
  `sample`.
- A non-blocking Polars CPU feature warning may appear in PM2 logs on VM24.
