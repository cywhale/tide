#!/usr/bin/env python
"""T-D2 (spec §7.4) — pyTMD 3.0.6 reference + gate. Run in the dev_tpxo10
env (`cd dev_tpxo10 && uv run python scripts/td2_golden.py`) AFTER
td2_ours.py has produced td2_ours.json in the production env.

For each station/window it predicts the elevation series from the TPXO10
SOURCE via pyTMD 3.0.6 `compute.tide_elevations` at the SAME instants as
ours (delta_time in SECONDS since the 1992 epoch), then gates the
2.2.8<->3.0.6 engine difference:

    per-station-window  RMSE <= 5 mm  AND  max|delta| <= 20 mm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import pyTMD.compute as C

REPO = Path(__file__).resolve().parents[2]

MODEL = "TPXO10-atlas-v2-nc"
EPOCH = (1992, 1, 1, 0, 0, 0)
RMSE_MM = 5.0
MAXABS_MM = 20.0


def reference_cm(lon, lat, t_days, source):
    t_sec = np.asarray(t_days, dtype=float) * 86400.0
    z = C.tide_elevations(np.array([lon]), np.array([lat]), t_sec,
                          directory=str(source), model=MODEL, epoch=EPOCH,
                          type="time series", crop=True, buffer=2)
    return np.asarray(z).ravel() * 100.0   # m -> cm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", type=Path,
                    default=REPO / "dev_tpxo10" / "benchmarks" / "td2_ours.json")
    ap.add_argument("--source", type=Path,
                    default=REPO / "data_src")
    ap.add_argument("--out", type=Path,
                    default=REPO / "dev_tpxo10" / "benchmarks" / "td2_result.json")
    args = ap.parse_args()

    ours = json.loads(args.ours.read_text())
    results = {"_meta": {"ours_store": ours["store"], "model": MODEL,
                         "gate": {"rmse_mm": RMSE_MM, "maxabs_mm": MAXABS_MM}},
               "stations": {}}
    failures = []
    worst_rmse = worst_max = 0.0
    for name, st in ours["stations"].items():
        lon, lat = st["lon"], st["lat"]
        entry = {}
        for w0, win in st["windows"].items():
            zc = np.array([np.nan if v is None else v for v in win["z_cm"]])
            if not np.all(np.isfinite(zc)):
                entry[w0] = {"skipped": "ours has missing values (invalid node)"}
                continue
            ref = reference_cm(lon, lat, win["t_days"], args.source)
            d_mm = np.abs(zc - ref) * 10.0          # cm -> mm
            rmse = float(np.sqrt(np.mean(((zc - ref) * 10.0) ** 2)))
            mx = float(d_mm.max())
            ok = rmse <= RMSE_MM and mx <= MAXABS_MM
            entry[w0] = {"rmse_mm": round(rmse, 3), "maxabs_mm": round(mx, 3),
                         "n": int(zc.size), "pass": ok}
            worst_rmse = max(worst_rmse, rmse); worst_max = max(worst_max, mx)
            if not ok:
                failures.append(f"{name}/{w0}: RMSE {rmse:.2f}mm max {mx:.2f}mm")
        results["stations"][name] = entry
        passes = [v.get("pass") for v in entry.values() if "pass" in v]
        tag = "PASS" if all(passes) else "FAIL"
        rs = ", ".join(f"{w}:rmse={v['rmse_mm']}mm/max={v['maxabs_mm']}mm"
                       for w, v in entry.items() if "rmse_mm" in v) or "(skipped)"
        print(f"  {name:20s} [{tag}] {rs}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(f"[summary] worst RMSE {worst_rmse:.3f} mm, worst max|d| {worst_max:.3f} mm "
          f"(gate {RMSE_MM}/{MAXABS_MM} mm)")
    if failures:
        print("FAIL td2_golden:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS td2_golden (T-D2 engine equivalence) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
