#!/usr/bin/env python
# coding: utf-8

"""
Case 3: Model validation / harmonic constants map (M2 amplitude).

Reproduces a core validation figure concept:
Co-amplitude chart (M2) for a regional domain, similar to comparisons in
"Accuracy assessment of global ocean tide models in the South China Sea using satellite altimeter and tide gauge data"
Acta Oceanologica Sinica (2020).

Method:
1) Query /api/tide/const for M2 amplitude over a grid.
2) Plot spatial distribution as a co-amplitude chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt


API_BASE = "https://eco.odb.ntu.edu.tw/api/tide/const"


def fetch_m2_amplitude(lons: np.ndarray, lats: np.ndarray) -> pd.DataFrame:
    lon_str = ",".join([f"{x:.3f}" for x in lons])
    lat_str = ",".join([f"{y:.3f}" for y in lats])

    params = {
        "lon": lon_str,
        "lat": lat_str,
        "constituent": "m2",
        "complex": "amp",
        "append": "z",
        "mode": "row",
    }

    resp = requests.get(API_BASE, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"API error: {resp.status_code} {resp.text[:200]}")

    data = resp.json()
    return pd.DataFrame(data)


def main():
    # Taiwan Strait / South China Sea subregion
    lon_range = np.arange(118.5, 122.6, 0.4)
    lat_range = np.arange(21.5, 26.1, 0.4)

    # Build grid and flatten for API call
    LON, LAT = np.meshgrid(lon_range, lat_range)
    lons = LON.ravel()
    lats = LAT.ravel()

    df = fetch_m2_amplitude(lons, lats)

    if df.empty:
        raise RuntimeError("No data returned from API.")

    # m2_amp is in cm (as returned by API)
    if "m2_amp" not in df.columns:
        raise RuntimeError("m2_amp not found in API response.")

    # Rebuild grid for plotting
    grid = df.pivot_table(index="latitude", columns="longitude", values="m2_amp")
    lon_grid, lat_grid = np.meshgrid(grid.columns.values, grid.index.values)

    plt.figure(figsize=(9, 8))
    levels = np.linspace(np.nanmin(grid.values), np.nanmax(grid.values), 12)
    cf = plt.contourf(lon_grid, lat_grid, grid.values, levels=levels, cmap="viridis", extend="both")
    cbar = plt.colorbar(cf)
    cbar.set_label("M2 Amplitude (cm)")

    cs = plt.contour(lon_grid, lat_grid, grid.values, levels=levels, colors="white", linewidths=0.6, alpha=0.7)
    plt.clabel(cs, inline=1, fontsize=8, fmt="%.1f")

    plt.title("M2 Co-Amplitude Chart (TPXO9-v5)\\nTaiwan Strait / Northern South China Sea")
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")
    plt.grid(True, linestyle=":", alpha=0.4)

    plt.figtext(
        0.5,
        0.01,
        "Extracted via ODB Tide API (/api/tide/const). Concept matches model validation charts in Acta Oceanol. Sin. (2020).",
        ha="center",
        fontsize=9,
        color="gray",
    )

    out_png = "examples/case3_model_validation_m2.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
