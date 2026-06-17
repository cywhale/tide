#!/usr/bin/env python
"""T-F observation validation harness (spec §7.6). Run in the PRODUCTION
env (`uv run --project <repo> ... tf_observation.py ...`).

STATUS: SCRIPT READY — GATE NOT EXECUTED. The prediction + scoring logic
is self-contained and unit-tested (`--self-test`), but the binding T-F
gate needs REAL tide-gauge observations supplied via `--observations
<file>`; it never hits the network itself (so it cannot be made flaky by
NOAA/CWA availability). Provide a sanitized observations file and re-run
to execute the gate.

Design: prediction (deterministic, both stores, local) is decoupled from
the observation source (parametrized file). For each station it predicts
the z elevation series at the OBSERVED instants from BOTH stores
(OLD = data/tpxo9.zarr legacy adapter, NEW = data/tpxo10.zarr) via the
runtime path, de-means model and obs (removes the datum/MSL offset),
and scores RMSE / bias / coverage per station.

Gate (spec §7.6): TPXO10 RMSE <= TPXO9 + 1 cm at >= 80% of stations AND
mean RMSE(TPXO10) <= mean RMSE(TPXO9). Stations where TPXO10 is worse by
> 3 cm are flagged for individual review.

Observations file format (JSON):
  {"<station_id>": {"lon": float, "lat": float,
                    "times_utc": ["YYYY-MM-DDTHH:MM:SS", ...],
                    "heights_cm": [float, ...]}, ...}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

EPOCH = datetime(1992, 1, 1, tzinfo=timezone.utc)
RMSE_TOL_CM = 1.0       # TPXO10 RMSE <= TPXO9 RMSE + 1 cm
PASS_FRACTION = 0.80    # at >= 80% of stations
REVIEW_WORSE_CM = 3.0   # individually review > 3 cm worse


def _days_since_epoch(times_utc):
    out = []
    for t in times_utc:
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        out.append((dt - EPOCH).total_seconds() / 86400.0)
    return np.asarray(out, dtype=float)


def predict_cm(adapter, lon, lat, t_days):
    """z elevation (cm) at the given instants via the runtime path."""
    from src.model_utils import get_tide_series   # noqa: E402
    sub = adapter.sel_point(lon, lat, tol=0.5 / 30.0)
    amp, ph = adapter.amp_ph(sub, "z")
    z = get_tide_series(np.ma.filled(amp, np.nan), np.ma.filled(ph, np.nan),
                        np.asarray(adapter.constituents), t_days,
                        unit="cm", drop_mask=True)
    return np.asarray(z).ravel()


def score_station(model_cm, obs_cm):
    """De-mean both (drop datum offset), score RMSE/bias/coverage on the
    finite-overlapping samples."""
    model = np.asarray(model_cm, float)
    obs = np.asarray(obs_cm, float)
    ok = np.isfinite(model) & np.isfinite(obs)
    n = int(ok.sum())
    if n < 3:
        return {"rmse_cm": None, "bias_cm": None, "coverage": n / max(len(obs), 1),
                "n": n}
    m, o = model[ok], obs[ok]
    bias = float(np.mean(m - o))           # datum offset (pre de-mean)
    md, od = m - m.mean(), o - o.mean()
    rmse = float(np.sqrt(np.mean((md - od) ** 2)))
    return {"rmse_cm": round(rmse, 3), "bias_cm": round(bias, 3),
            "coverage": round(n / len(obs), 3), "n": n}


def run_gate(observations: dict, old_adapter, new_adapter) -> dict:
    stations = {}
    old_rmse, new_rmse = [], []
    pass_count = total = 0
    review = []
    for sid, st in observations.items():
        t_days = _days_since_epoch(st["times_utc"])
        obs = np.asarray(st["heights_cm"], float)
        so = score_station(predict_cm(old_adapter, st["lon"], st["lat"], t_days), obs)
        sn = score_station(predict_cm(new_adapter, st["lon"], st["lat"], t_days), obs)
        rec = {"lon": st["lon"], "lat": st["lat"], "old": so, "new": sn}
        if so["rmse_cm"] is not None and sn["rmse_cm"] is not None:
            total += 1
            old_rmse.append(so["rmse_cm"]); new_rmse.append(sn["rmse_cm"])
            within = sn["rmse_cm"] <= so["rmse_cm"] + RMSE_TOL_CM
            rec["within_tol"] = within
            pass_count += int(within)
            if sn["rmse_cm"] > so["rmse_cm"] + REVIEW_WORSE_CM:
                review.append(sid)
        stations[sid] = rec
    frac = pass_count / total if total else 0.0
    mean_old = float(np.mean(old_rmse)) if old_rmse else None
    mean_new = float(np.mean(new_rmse)) if new_rmse else None
    gate_pass = (total > 0 and frac >= PASS_FRACTION
                 and mean_new is not None and mean_new <= mean_old)
    return {"stations": stations,
            "summary": {"n_stations": total,
                        "frac_within_1cm": round(frac, 3),
                        "mean_rmse_old_cm": None if mean_old is None else round(mean_old, 3),
                        "mean_rmse_new_cm": None if mean_new is None else round(mean_new, 3),
                        "stations_to_review_gt3cm": review,
                        "gate_pass": gate_pass}}


def _self_test_observations(new_adapter):
    """Synthetic obs = the tpxo10 model's own prediction + small noise at
    a few coastal points, to exercise the harness end-to-end with NO
    network. NOT a validation gate — proves the scoring/gate plumbing."""
    rng = np.random.default_rng(1)
    pts = {"syn_taiwan": (119.5, 24.5), "syn_pacific": (140.0, 20.0)}
    base = datetime(2023, 7, 25, tzinfo=timezone.utc)
    times = [(base.replace(hour=h)).strftime("%Y-%m-%dT%H:%M:%S") for h in range(24)]
    t_days = _days_since_epoch(times)
    obs = {}
    for sid, (lon, lat) in pts.items():
        z = predict_cm(new_adapter, lon, lat, t_days)
        obs[sid] = {"lon": lon, "lat": lat, "times_utc": times,
                    "heights_cm": (z + rng.normal(0, 0.3, z.size) + 50.0).tolist()}
    return obs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observations", type=Path,
                    help="JSON of station tide-gauge observations (REAL gate input)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the harness on synthetic obs (no network; plumbing only)")
    ap.add_argument("--old", type=Path, default=REPO / "data" / "tpxo9.zarr")
    ap.add_argument("--new", type=Path, default=REPO / "data" / "tpxo10.zarr")
    ap.add_argument("--out", type=Path,
                    default=REPO / "dev_tpxo10" / "benchmarks" / "tf_result.json")
    args = ap.parse_args()

    # no-args: print the status and return BEFORE opening any store (round
    # 26 F1) — "SCRIPT READY, GATE NOT EXECUTED" must not require a store.
    if not args.self_test and not args.observations:
        print("T-F harness: SCRIPT READY, GATE NOT EXECUTED.\n"
              "Provide --observations <file> (real tide-gauge data) to run the\n"
              "binding gate, or --self-test to exercise the plumbing.")
        return 0

    from src import store_adapter as SA   # noqa: E402
    old_fallback = False
    if args.observations:
        # binding gate: BOTH stores must exist — no silent old->new fallback
        # (round 26 F2), which would compare TPXO10 against itself.
        missing = [str(p) for p in (args.old, args.new) if not p.exists()]
        if missing:
            print(f"ERROR: --observations gate requires both stores; missing "
                  f"{missing}. The TPXO9 baseline and TPXO10 store must both "
                  "be present to compare. Aborting (no fallback).")
            return 2
        new = SA.open_store(str(args.new))
        old = SA.open_store(str(args.old))
        obs = json.loads(args.observations.read_text())
        mode = f"GATE ({args.observations})"
        # the default --out (tf_result.json) is gitignored as self-test
        # output; binding gate evidence must be saved to a TRACKED path so
        # it survives into the G3 close-out (reviewer round 26 note).
        if args.out.name == "tf_result.json":
            print("WARNING: binding T-F gate is writing to the gitignored "
                  "default tf_result.json. Re-run with an explicit tracked "
                  "--out, e.g. --out dev_tpxo10/benchmarks/"
                  "tf_observation_real_YYYYMMDD.json, to preserve G3 evidence.")
    else:  # self-test: fallback to new is allowed but flagged
        new = SA.open_store(str(args.new))
        if args.old.exists():
            old = SA.open_store(str(args.old))
        else:
            old = new; old_fallback = True
        obs = _self_test_observations(new)
        mode = "SELF-TEST (synthetic obs; plumbing only — NOT the T-F gate)"

    result = run_gate(obs, old, new)
    result["_meta"] = {"mode": mode, "old": str(args.old), "new": str(args.new),
                       "old_fallback_to_new": old_fallback}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    sm = result["summary"]
    print(f"[{mode}]")
    for sid, r in result["stations"].items():
        o, n = r["old"]["rmse_cm"], r["new"]["rmse_cm"]
        print(f"  {sid:16s} OLD rmse {o} cm  NEW rmse {n} cm  "
              f"cov {r['new']['coverage']}")
    print(f"[summary] {sm['n_stations']} stations | within 1cm: "
          f"{sm['frac_within_1cm']} | mean RMSE old {sm['mean_rmse_old_cm']} "
          f"new {sm['mean_rmse_new_cm']} cm | gate_pass={sm['gate_pass']}")
    if args.self_test:
        print("NOTE: self-test only — the binding T-F gate requires real obs.")
        return 0
    return 0 if sm["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
