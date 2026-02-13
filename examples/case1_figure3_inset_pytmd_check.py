#!/usr/bin/env python
# coding: utf-8

"""
Validate Figure 3 inset timing using pyTMD vs ODB Tide API.

This script:
1) Extracts harmonic constants (amp/phase) from TPXO ATLAS netCDF files.
2) Uses pyTMD to predict sea surface elevation.
3) Fetches ODB Tide API time series at the same point.
4) Compares phase/timing and prints differences.

Notes:
- ODB Tide API returns hourly series (cm). pyTMD can do finer (30-min).
- Figure 3 in the paper reports specific phase times (e.g., 09:30, 11:00, 15:30, 17:00 UTC).
- Provide TPXO ATLAS netCDF files via:
  - env: TPXO_ATLAS_DIR, TPXO_GRID_FILE, TPXO_MODEL_GLOB
  - or CLI: --tpxo-dir, --grid-file, --model-glob
"""

from __future__ import annotations

from datetime import datetime, timedelta
import argparse

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pyTMD.predict
import pyTMD.io.ATLAS as ATLAS
import timescale.time


API_TIDE = "https://eco.odb.ntu.edu.tw/api/tide"

# Pick a point in open water of Taiwan Strait
LON, LAT = 120.0, 23.0

# Time window (UTC/GMT)
START = datetime(2013, 6, 27, 0, 0, 0)
END = datetime(2013, 6, 28, 23, 0, 0)

# Paper phase markers (UTC)
PAPER_PHASES = [
    ("A 09:30", datetime(2013, 6, 27, 9, 30)),
    ("B 11:00", datetime(2013, 6, 27, 11, 0)),
    ("C 15:30", datetime(2013, 6, 27, 15, 30)),
    ("D 17:00", datetime(2013, 6, 27, 17, 0)),
]


def build_time_index(start: datetime, end: datetime, step_minutes: int = 60):
    times = []
    t = start
    while t <= end:
        times.append(t)
        t += timedelta(minutes=step_minutes)
    return times


def tide_time_from_datetime(times):
    # pyTMD uses days since 1992-01-01
    out = []
    for t in times:
        out.append(timescale.time.convert_calendar_dates(t.year, t.month, t.day, t.hour, t.minute, t.second))
    return np.array(out)


def fetch_odb_series(lon: float, lat: float, start: datetime, end: datetime):
    params = {
        "lon0": lon,
        "lat0": lat,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "append": "z",
        "sample": 1,
    }
    resp = requests.get(API_TIDE, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"API error: {resp.status_code} {resp.text[:200]}")
    data = resp.json()
    if not data:
        raise RuntimeError("Empty response from ODB API for this point.")
    times = [datetime.fromisoformat(t) for t in data["time"]]
    z_m = np.array(data["z"], dtype=float) / 100.0  # cm -> m
    return times, z_m


def predict_pytmd_series(lon: float, lat: float, times: list[datetime], model_dir: str | None, grid_file: str | None, model_glob: str | None):
    import glob
    from pathlib import Path

    if not model_dir and not grid_file:
        raise RuntimeError(
            "TPXO ATLAS netCDF files are required. "
            "Set TPXO_ATLAS_DIR (directory containing grid/model files) "
            "or TPXO_GRID_FILE (full path to grid file)."
        )

    model_dir_path = Path(model_dir).expanduser() if model_dir else None

    if grid_file:
        grid_path = Path(grid_file).expanduser()
    else:
        grid_candidates = sorted(model_dir_path.glob("grid_*tpxo9_atlas*"))
        if not grid_candidates:
            grid_candidates = sorted(model_dir_path.glob("grid_*tpxo9*"))
        if not grid_candidates:
            raise RuntimeError("No grid file found (grid_*tpxo9_atlas*).")
        grid_path = grid_candidates[0]

    if model_glob:
        model_files = sorted(glob.glob(model_glob))
    else:
        model_files = sorted(model_dir_path.glob("h_*tpxo9_atlas*"))
        if not model_files:
            model_files = sorted(model_dir_path.glob("z_*tpxo9_atlas*"))
        if not model_files:
            model_files = sorted(model_dir_path.glob("h_*tpxo9*"))
        if not model_files:
            model_files = sorted(model_dir_path.glob("z_*tpxo9*"))
        if not model_files:
            raise RuntimeError("No model files found (h_*tpxo9_atlas* or z_*tpxo9_atlas*).")

    # Detect compression based on file extension
    compressed = any(str(p).endswith(".gz") for p in [grid_path] + list(model_files))

    # Crop to a small bounding box to speed up reading
    bounds = [lon - 1.0, lon + 1.0, lat - 1.0, lat + 1.0]
    amp, ph, _, cons = ATLAS.extract_constants(
        np.array([lon]),
        np.array([lat]),
        grid_file=grid_path,
        model_files=[str(p) for p in model_files],
        type="z",
        compressed=compressed,
        method="bilinear",
        crop=True,
        bounds=bounds,
        scale=1e-3,
        extrapolate=False,
    )

    amp = np.atleast_2d(amp).astype(float)
    ph = np.atleast_2d(ph).astype(float)
    cons = [str(c) for c in cons]

    # Complex harmonic constants
    hc = (amp * np.exp(-1j * ph * np.pi / 180.0))
    hc = np.ma.array(hc, mask=np.isnan(hc))

    tide_time = tide_time_from_datetime(times)
    deltat = np.zeros_like(tide_time)

    tide = pyTMD.predict.time_series(tide_time, hc, cons, deltat=deltat, corrections="OTIS")
    minor = pyTMD.predict.infer_minor(tide_time, hc, cons, deltat=deltat, corrections="OTIS")
    tide.data[:] += minor.data[:]
    out = np.array(tide.data)
    if np.ma.isMaskedArray(out):
        out = out.filled(np.nan)
    return out


def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return np.sqrt(np.nanmean((a[m] - b[m]) ** 2))


def phase_lag_hours(a, b):
    # Cross-correlation lag: positive means b lags a
    m = np.isfinite(a) & np.isfinite(b)
    a0 = a[m] - np.nanmean(a[m])
    b0 = b[m] - np.nanmean(b[m])
    c = np.correlate(a0, b0, mode="full")
    lag = np.argmax(c) - (len(a0) - 1)
    return lag  # in samples (hours if hourly data)


def max_min_times(times, values, day):
    day_mask = np.array([t.date() == day.date() for t in times])
    tday = np.array(times)[day_mask]
    vday = np.array(values)[day_mask]
    if len(vday) == 0:
        return None, None
    max_t = tday[np.nanargmax(vday)]
    min_t = tday[np.nanargmin(vday)]
    return max_t, min_t


def main():
    parser = argparse.ArgumentParser(description="Compare ODB Tide API with pyTMD using TPXO ATLAS netCDF files.")
    parser.add_argument("--tpxo-dir", default=None, help="Directory containing TPXO ATLAS netCDF files")
    parser.add_argument("--grid-file", default=None, help="Full path to ATLAS grid file")
    parser.add_argument("--model-glob", default=None, help="Glob for model files (e.g., 'h_*tpxo9_atlas_30_v5*.nc*')")
    args = parser.parse_args()

    import os
    model_dir = args.tpxo_dir or os.getenv("TPXO_ATLAS_DIR")
    grid_file = args.grid_file or os.getenv("TPXO_GRID_FILE")
    model_glob = args.model_glob or os.getenv("TPXO_MODEL_GLOB")

    # Hourly series to compare with ODB API
    t_hourly = build_time_index(START, END, step_minutes=60)
    odb_times, odb_z = fetch_odb_series(LON, LAT, START, END)
    pytmd_z_hourly = predict_pytmd_series(LON, LAT, t_hourly, model_dir, grid_file, model_glob)

    # 30-min series for paper phase comparison
    t_30 = build_time_index(START, END, step_minutes=30)
    pytmd_z_30 = predict_pytmd_series(LON, LAT, t_30, model_dir, grid_file, model_glob)

    # Compare ODB vs pyTMD (hourly)
    if len(odb_times) != len(t_hourly):
        print("Warning: ODB API returned different time length than expected.")
    n = min(len(odb_z), len(pytmd_z_hourly))
    odb_z = odb_z[:n]
    pytmd_z_hourly = pytmd_z_hourly[:n]

    print("ODB vs pyTMD (hourly) comparison")
    print("RMSE (m):", rmse(odb_z, pytmd_z_hourly))
    print("Mean bias (m):", np.nanmean(odb_z - pytmd_z_hourly))
    print("Max abs diff (m):", np.nanmax(np.abs(odb_z - pytmd_z_hourly)))
    print("Lag (hours, positive means pyTMD lags ODB):", phase_lag_hours(odb_z, pytmd_z_hourly))

    # Compare to paper timing (pyTMD 30-min)
    max_t, min_t = max_min_times(t_30, pytmd_z_30, START)
    print("pyTMD 30-min max on 2013-06-27:", max_t)
    print("pyTMD 30-min min on 2013-06-27:", min_t)
    print("Paper phase times (UTC):")
    for label, t in PAPER_PHASES:
        print(f"  {label}: {t}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(odb_times[:n], odb_z, label="ODB Tide API (hourly)", linewidth=1.2)
    ax.plot(t_hourly[:n], pytmd_z_hourly, label="pyTMD (hourly)", linewidth=1.2, linestyle="--")

    for label, t in PAPER_PHASES:
        ax.axvline(t, color="gray", linestyle=":", linewidth=0.8)
        ax.text(t, ax.get_ylim()[1] * 0.9, label, rotation=90, fontsize=8, color="gray")

    ax.set_title("Inset SSE: ODB Tide API vs pyTMD (TPXO9)\\n2013-06-27 to 2013-06-28")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Sea Surface Elevation (m)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_png = "examples/case1_figure3_inset_pytmd_check.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
