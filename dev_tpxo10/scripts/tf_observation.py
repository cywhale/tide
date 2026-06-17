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
> 3 cm are flagged for individual review WITH a dominant-cause label
(amplitude vs phase-lag) from the report-only diagnostics below.

Report-only diagnostics (round 27, do NOT change the binding gate): per
station, for BOTH stores, a phase/shape/amplitude profile —
best-lag normalized correlation (within +/-180 min), corr at zero/best
lag, de-meaned shape RMSE at best lag, and amplitude std/range ratios.
This captures "amplitude may differ but phase mostly agrees", lets a
RMSE regression be triaged, and shows whether TPXO10's phase/shape
regresses vs TPXO9. Review policy (spec §7.6): if the RMSE gate fails
but the failures concentrate on known amplitude-sensitive stations AND
TPXO10's phase/shape does not regress, escalate to manual review rather
than auto-failing the data upgrade.

Data-source hygiene: this harness NEVER fetches (it reads --observations,
decoupled from the network). If a CWA/NOAA fetcher is added later, a CWA
token may be read from `.env` but MUST NOT be written to any output or
log; NOAA needs no token and supports a longer, more stable window
(suited as the primary gate; CWA 24 h is a recent-window smoke).

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


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    da, db = np.sqrt(np.sum(a*a)), np.sqrt(np.sum(b*b))
    if da == 0 or db == 0:
        return float("nan")
    return float(np.sum(a*b) / (da*db))


def phase_amplitude_diagnostics(model_cm, obs_cm, dt_minutes, max_lag_min=180.0):
    """REPORT-ONLY (round 27): phase/shape/amplitude diagnostics on the
    finite-overlapping, de-meaned series. Captures the "amplitude may
    differ but phase mostly agrees" pattern the owner flagged:

      corr_at_zero_lag : normalized correlation, no shift
      best_lag_min     : lag (min, within +/-max_lag_min) maximizing corr;
                         positive = model LEADS obs
      corr_at_best_lag : correlation at that lag
      shape_rmse_cm    : de-meaned RMSE after aligning at best_lag
      std_ratio        : std(model)/std(obs)  (amplitude)
      range_ratio      : (max-min model)/(max-min obs)
    """
    m = np.asarray(model_cm, float); o = np.asarray(obs_cm, float)
    ok = np.isfinite(m) & np.isfinite(o)
    if int(ok.sum()) < 4 or not np.isfinite(dt_minutes) or dt_minutes <= 0:
        return None
    m, o = m[ok], o[ok]
    md, od = m - m.mean(), o - o.mean()
    corr0 = _pearson(md, od)
    max_lag = int(round(max_lag_min / dt_minutes))
    best_lag, best_corr = 0, corr0 if np.isfinite(corr0) else -2.0
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            a, b = md[k:], od[:len(od)-k] if k > 0 else od
        else:
            a, b = md[:len(md)+k], od[-k:]
        if len(a) < 4:
            continue
        c = _pearson(a, b)
        if np.isfinite(c) and c > best_corr:
            best_corr, best_lag = c, k
    # shape RMSE at best lag (aligned, re-de-meaned on the overlap)
    if best_lag >= 0:
        a, b = md[best_lag:], od[:len(od)-best_lag] if best_lag > 0 else od
    else:
        a, b = md[:len(md)+best_lag], od[-best_lag:]
    a, b = a - a.mean(), b - b.mean()
    shape_rmse = float(np.sqrt(np.mean((a - b) ** 2))) if len(a) >= 4 else None
    so, ss = float(o.std()), float(m.std())
    rng_o = float(o.max() - o.min()); rng_m = float(m.max() - m.min())
    return {
        "corr_at_zero_lag": None if not np.isfinite(corr0) else round(corr0, 4),
        "best_lag_min": round(best_lag * dt_minutes, 1),
        "corr_at_best_lag": None if not np.isfinite(best_corr) else round(best_corr, 4),
        "shape_rmse_cm": None if shape_rmse is None else round(shape_rmse, 3),
        "std_ratio": None if so == 0 else round(ss / so, 4),
        "range_ratio": None if rng_o == 0 else round(rng_m / rng_o, 4),
    }


def classify_failure(diag):
    """Label a worse-than-tolerance station as amplitude- vs phase-lag-
    dominated (report-only triage; round 27)."""
    if not diag:
        return "unknown"
    lag = abs(diag.get("best_lag_min") or 0.0)
    sr = diag.get("std_ratio")
    amp_off = sr is not None and abs(sr - 1.0) >= 0.15
    phase_off = lag >= 30.0
    if amp_off and not phase_off:
        return "amplitude"
    if phase_off and not amp_off:
        return "phase-lag"
    if amp_off and phase_off:
        return "mixed"
    return "within-shape"   # neither amplitude nor phase explains it


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
        dt_min = float(np.median(np.diff(t_days)) * 1440.0) if len(t_days) > 1 else float("nan")
        m_old = predict_cm(old_adapter, st["lon"], st["lat"], t_days)
        m_new = predict_cm(new_adapter, st["lon"], st["lat"], t_days)
        so = score_station(m_old, obs)
        sn = score_station(m_new, obs)
        # report-only phase/shape/amplitude diagnostics for BOTH stores so a
        # RMSE regression can be triaged (amplitude vs phase) and TPXO10
        # phase non-regression vs TPXO9 shown (round 27)
        diag_old = phase_amplitude_diagnostics(m_old, obs, dt_min)
        diag_new = phase_amplitude_diagnostics(m_new, obs, dt_min)
        rec = {"lon": st["lon"], "lat": st["lat"], "old": so, "new": sn,
               "diag_old": diag_old, "diag_new": diag_new}
        if so["rmse_cm"] is not None and sn["rmse_cm"] is not None:
            total += 1
            old_rmse.append(so["rmse_cm"]); new_rmse.append(sn["rmse_cm"])
            within = sn["rmse_cm"] <= so["rmse_cm"] + RMSE_TOL_CM
            rec["within_tol"] = within
            pass_count += int(within)
            if sn["rmse_cm"] > so["rmse_cm"] + REVIEW_WORSE_CM:
                review.append({"station": sid,
                               "old_rmse_cm": so["rmse_cm"], "new_rmse_cm": sn["rmse_cm"],
                               "dominant": classify_failure(diag_new),
                               "new_best_lag_min": (diag_new or {}).get("best_lag_min"),
                               "new_std_ratio": (diag_new or {}).get("std_ratio")})
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
        dn = r.get("diag_new") or {}
        print(f"  {sid:16s} OLD rmse {o} cm  NEW rmse {n} cm  "
              f"cov {r['new']['coverage']} | NEW lag {dn.get('best_lag_min')}min "
              f"corr0 {dn.get('corr_at_zero_lag')} std_ratio {dn.get('std_ratio')}")
    if sm["stations_to_review_gt3cm"]:
        print("  [review >3cm-worse]")
        for rv in sm["stations_to_review_gt3cm"]:
            print(f"    {rv['station']}: old {rv['old_rmse_cm']} -> new "
                  f"{rv['new_rmse_cm']} cm, dominant={rv['dominant']} "
                  f"(lag {rv['new_best_lag_min']}min, std_ratio {rv['new_std_ratio']})")
    print(f"[summary] {sm['n_stations']} stations | within 1cm: "
          f"{sm['frac_within_1cm']} | mean RMSE old {sm['mean_rmse_old_cm']} "
          f"new {sm['mean_rmse_new_cm']} cm | gate_pass={sm['gate_pass']}")
    if args.self_test:
        print("NOTE: self-test only — the binding T-F gate requires real obs.")
        return 0
    return 0 if sm["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
