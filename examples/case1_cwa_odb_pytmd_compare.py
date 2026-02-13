#!/usr/bin/env python
# coding: utf-8

"""
Compare CWA observed tide height with ODB Tide API and pyTMD (TPXO9 ATLAS).

Goal:
- Use recent 24-hour CWA tide-gauge observations near Taiwan Strait (paper domain).
- Convert CWA local timestamps (+08:00) to UTC.
- Compare against ODB Tide API and pyTMD at the same coordinate and UTC timestamps.
- Quantify amplitude and phase agreement to support phase correctness.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy import signal

import pyTMD.io.ATLAS as ATLAS
import pyTMD.predict
import timescale.time


API_CWA = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-B0075-002"
API_ODB = "https://eco.odb.ntu.edu.tw/api/tide"
STATIONS_JSON = "test/stations_cwa.json"
DEFAULT_TPXO_DIR = "data_src/TPXO9_atlas_v5"

# Taiwan Strait / near paper domain
PAPER_BBOX = (117.0, 123.0, 20.0, 27.0)  # lon_min, lon_max, lat_min, lat_max


def cwa_local_start_24h() -> str:
    tw = ZoneInfo("Asia/Taipei")
    now_local = datetime.now(tz=tw)
    start_local = (now_local - timedelta(hours=24)).replace(minute=0, second=0, microsecond=0)
    return start_local.strftime("%Y-%m-%dT%H:%M:%S")


def load_cwa_stations(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    locs = data["cwaopendata"]["Resources"]["Resource"]["Data"]["SeaSurfaceObs"]["Location"]
    rows = []
    for loc in locs:
        st = loc["Station"]
        rows.append(
            {
                "station_id": st.get("StationID"),
                "station_name": st.get("StationName", ""),
                "station_attr": st.get("StationAttribute", ""),
                "lon": float(st.get("StationLongitude")),
                "lat": float(st.get("StationLatitude")),
            }
        )
    return pd.DataFrame(rows)


def select_stations_in_domain(stations: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    lon_min, lon_max, lat_min, lat_max = bbox
    m = (
        (stations["lon"] >= lon_min)
        & (stations["lon"] <= lon_max)
        & (stations["lat"] >= lat_min)
        & (stations["lat"] <= lat_max)
        & (stations["station_attr"].str.contains("潮位站", na=False))
    )
    return stations[m].copy()


def fetch_cwa_station_observed(station_id: str, token: str, local_time_from: str) -> pd.DataFrame:
    params = {
        "Authorization": token,
        "StationID": station_id,
        "WeatherElement": "TideHeight,TideLevel",
        "sort": "DataTime",
        "timeFrom": local_time_from,
    }
    r = requests.get(API_CWA, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    locations = data.get("Records", {}).get("SeaSurfaceObs", {}).get("Location", [])
    if not locations:
        return pd.DataFrame()

    obs = locations[0].get("StationObsTimes", {}).get("StationObsTime", [])
    rows = []
    for rec in obs:
        tide_h = rec.get("WeatherElements", {}).get("TideHeight")
        if tide_h in (None, "None", "-", ""):
            continue
        t_local = pd.to_datetime(rec["DateTime"], utc=True)
        rows.append(
            {
                "timestamp_utc": t_local,
                "height_cwa_m": float(tide_h),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"])
    return df


def fetch_odb_series(lon: float, lat: float, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    params = {
        "lon0": lon,
        "lat0": lat,
        "start": start_utc.strftime("%Y-%m-%dT%H:%M:%S"),
        "end": end_utc.strftime("%Y-%m-%dT%H:%M:%S"),
        "append": "z",
        "sample": 1,
    }
    r = requests.get(API_ODB, params=params, timeout=60)
    r.raise_for_status()
    d = r.json()
    if not d:
        return pd.DataFrame()
    t = pd.to_datetime(d["time"], utc=True)
    z_m = np.array(d["z"], dtype=float) / 100.0
    return pd.DataFrame({"timestamp_utc": t, "height_odb_m": z_m})


def tide_time_from_datetime(times_utc: pd.Series) -> np.ndarray:
    out = []
    for t in times_utc:
        tt = t.to_pydatetime()
        out.append(timescale.time.convert_calendar_dates(tt.year, tt.month, tt.day, tt.hour, tt.minute, tt.second))
    return np.array(out)


def extract_station_constants(lon: float, lat: float, tpxo_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model_dir = Path(tpxo_dir)
    grid_candidates = sorted(model_dir.glob("grid_*tpxo9_atlas*"))
    if not grid_candidates:
        raise RuntimeError(f"No TPXO grid file found in {tpxo_dir}")
    grid_file = grid_candidates[0]
    model_files = sorted(model_dir.glob("h_*tpxo9_atlas*"))
    if not model_files:
        raise RuntimeError(f"No TPXO model files (h_*) found in {tpxo_dir}")

    amp, ph, _, cons = ATLAS.extract_constants(
        np.array([lon]),
        np.array([lat]),
        grid_file=str(grid_file),
        model_files=[str(x) for x in model_files],
        type="z",
        compressed=False,
        method="bilinear",
        crop=True,
        bounds=[lon - 1.0, lon + 1.0, lat - 1.0, lat + 1.0],
        scale=1e-3,
        extrapolate=False,
    )
    return np.atleast_2d(amp).astype(float), np.atleast_2d(ph).astype(float), [str(c) for c in cons]


def predict_pytmd_at_times(amp: np.ndarray, ph: np.ndarray, cons: list[str], times_utc: pd.Series) -> np.ndarray:
    hc = amp * np.exp(-1j * ph * np.pi / 180.0)
    hc = np.ma.array(hc, mask=np.isnan(hc))
    tide_time = tide_time_from_datetime(times_utc)
    deltat = np.zeros_like(tide_time)
    tide = pyTMD.predict.time_series(tide_time, hc, cons, deltat=deltat, corrections="OTIS")
    minor = pyTMD.predict.infer_minor(tide_time, hc, cons, deltat=deltat, corrections="OTIS")
    tide.data[:] += minor.data[:]
    out = np.array(tide.data)
    if np.ma.isMaskedArray(out):
        out = out.filled(np.nan)
    return out


def phase_lag_hours(ref: np.ndarray, test: np.ndarray) -> float:
    m = np.isfinite(ref) & np.isfinite(test)
    if m.sum() < 3:
        return np.nan
    x = ref[m] - np.nanmean(ref[m])
    y = test[m] - np.nanmean(test[m])
    corr = signal.correlate(y, x, mode="full")
    lags = signal.correlation_lags(len(y), len(x), mode="full")
    lag = lags[np.argmax(corr)]  # samples (1 hour)
    return float(lag)


def summarize_pair(df: pd.DataFrame, target_col: str) -> dict:
    x = df["height_cwa_m"].to_numpy()
    y = df[target_col].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return {
            "n": int(m.sum()),
            "bias_m": np.nan,
            "rmse_m": np.nan,
            "corr": np.nan,
            "lag_hours": np.nan,
            "std_ratio": np.nan,
        }
    x = x[m]
    y = y[m]
    return {
        "n": int(len(x)),
        "bias_m": float(np.mean(y - x)),
        "rmse_m": float(np.sqrt(np.mean((y - x) ** 2))),
        "corr": float(np.corrcoef(x, y)[0, 1]),
        "lag_hours": phase_lag_hours(x, y),
        "std_ratio": float(np.std(y) / np.std(x)) if np.std(x) > 0 else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare CWA observed tide with ODB and pyTMD.")
    parser.add_argument("--tpxo-dir", default=DEFAULT_TPXO_DIR, help="TPXO9 ATLAS directory")
    parser.add_argument("--max-stations", type=int, default=10, help="Maximum number of stations to evaluate")
    parser.add_argument("--out-prefix", default="examples/case1_cwa_odb_pytmd", help="Output prefix")
    args = parser.parse_args()

    load_dotenv(".env")
    cwa_token = os.getenv("CWA_TOKEN")
    if not cwa_token:
        raise RuntimeError("CWA_TOKEN is missing in .env")

    local_start = cwa_local_start_24h()
    stations = load_cwa_stations(STATIONS_JSON)
    candidates = select_stations_in_domain(stations, PAPER_BBOX)
    print(f"Candidate tide stations in/near paper domain: {len(candidates)}")
    print("Local timeFrom (Asia/Taipei):", local_start)

    station_rows = []
    timeseries_rows = []
    used = 0

    for _, st in candidates.iterrows():
        if used >= args.max_stations:
            break
        sid = st["station_id"]
        sname = st["station_name"]
        lon = float(st["lon"])
        lat = float(st["lat"])

        cwa_df = fetch_cwa_station_observed(sid, cwa_token, local_start)
        if cwa_df.empty or len(cwa_df) < 8:
            continue

        t0 = cwa_df["timestamp_utc"].min()
        t1 = cwa_df["timestamp_utc"].max()
        odb_df = fetch_odb_series(lon, lat, t0, t1)
        if odb_df.empty:
            continue

        amp, ph, cons = extract_station_constants(lon, lat, args.tpxo_dir)
        py_vals = predict_pytmd_at_times(amp, ph, cons, cwa_df["timestamp_utc"])
        py_df = pd.DataFrame({"timestamp_utc": cwa_df["timestamp_utc"], "height_pytmd_m": py_vals})

        merged = cwa_df.merge(odb_df, on="timestamp_utc", how="inner").merge(py_df, on="timestamp_utc", how="inner")
        if merged.empty or len(merged) < 8:
            continue

        odb_stat = summarize_pair(merged, "height_odb_m")
        py_stat = summarize_pair(merged, "height_pytmd_m")

        station_rows.append(
            {
                "station_id": sid,
                "station_name": sname,
                "lon": lon,
                "lat": lat,
                "n": int(len(merged)),
                "odb_bias_m": odb_stat["bias_m"],
                "odb_rmse_m": odb_stat["rmse_m"],
                "odb_corr": odb_stat["corr"],
                "odb_lag_hours": odb_stat["lag_hours"],
                "odb_std_ratio": odb_stat["std_ratio"],
                "py_bias_m": py_stat["bias_m"],
                "py_rmse_m": py_stat["rmse_m"],
                "py_corr": py_stat["corr"],
                "py_lag_hours": py_stat["lag_hours"],
                "py_std_ratio": py_stat["std_ratio"],
            }
        )

        block = merged.copy()
        block["station_id"] = sid
        block["station_name"] = sname
        block["lon"] = lon
        block["lat"] = lat
        timeseries_rows.append(block)

        used += 1
        print(f"Processed {sid} {sname}: n={len(merged)}, ODB lag={odb_stat['lag_hours']}h, pyTMD lag={py_stat['lag_hours']}h")

    if not station_rows:
        raise RuntimeError("No usable stations found with recent CWA TideHeight data in the selected domain.")

    summary_df = pd.DataFrame(station_rows).sort_values(["odb_rmse_m", "py_rmse_m"])
    ts_df = pd.concat(timeseries_rows, ignore_index=True)

    out_csv = f"{args.out_prefix}_summary.csv"
    out_ts_csv = f"{args.out_prefix}_timeseries.csv"
    summary_df.to_csv(out_csv, index=False)
    ts_df.to_csv(out_ts_csv, index=False)
    print("Saved:", out_csv)
    print("Saved:", out_ts_csv)

    # Plot first 4 stations for visual phase check
    top = summary_df.head(4)["station_id"].tolist()
    fig, axes = plt.subplots(len(top), 1, figsize=(12, 2.6 * len(top)), sharex=True)
    if len(top) == 1:
        axes = [axes]

    for ax, sid in zip(axes, top):
        d = ts_df[ts_df["station_id"] == sid].sort_values("timestamp_utc")
        name = d["station_name"].iloc[0]
        ax.plot(d["timestamp_utc"], d["height_cwa_m"], label="CWA observed", linewidth=1.4, color="k")
        ax.plot(d["timestamp_utc"], d["height_odb_m"], label="ODB TPXO9", linewidth=1.2, color="tab:blue")
        ax.plot(d["timestamp_utc"], d["height_pytmd_m"], label="pyTMD TPXO9", linewidth=1.2, color="tab:orange", linestyle="--")
        ax.set_title(f"{sid} {name}")
        ax.set_ylabel("Tide Height (m)")
        ax.grid(True, linestyle=":", alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=timezone.utc))
    axes[-1].set_xlabel("UTC")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    plt.tight_layout()
    out_png = f"{args.out_prefix}_plot.png"
    plt.savefig(out_png, dpi=160)
    print("Saved:", out_png)

    print("\nSummary (key columns):")
    show_cols = ["station_id", "station_name", "n", "odb_rmse_m", "odb_corr", "odb_lag_hours", "py_rmse_m", "py_corr", "py_lag_hours"]
    print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
