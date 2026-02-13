#!/usr/bin/env python
# coding: utf-8

"""
Figure 4 style tidal current vector maps using ODB Tide API (TPXO9-atlas-v5).

This script fetches barotropic tidal current components (u, v) at four UTC phases
and renders 2x2 vector maps over Taiwan Strait.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


API_TIDE = "https://eco.odb.ntu.edu.tw/api/tide"

# Match Figure 4 framing more closely
LON0, LON1 = 116.8, 120.8
LAT0, LAT1 = 23.0, 25.0

# Use finest API grid; keep vectors dense to reflect 1/30 deg mesh
API_SAMPLE = 1
QUIVER_STRIDE = 1

PHASES = [
    ("A", "2013-06-27T04:30:00"),
    ("B", "2013-06-27T06:00:00"),  # max flood
    ("C", "2013-06-27T10:30:00"),
    ("D", "2013-06-27T12:00:00"),  # max ebb
]


def fetch_uv_map(when: str) -> dict:
    params = {
        "lon0": LON0,
        "lon1": LON1,
        "lat0": LAT0,
        "lat1": LAT1,
        "start": when,
        "append": "u,v",
        "sample": API_SAMPLE,
    }
    r = requests.get(API_TIDE, params=params, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"API error {r.status_code}: {r.text[:180]}")
    out = r.json()
    if not out:
        raise RuntimeError("Empty response from /api/tide for uv map.")
    return out


def uv_to_grid(data: dict):
    df = pd.DataFrame(
        {
            "lon": data["longitude"],
            "lat": data["latitude"],
            "u_cm_s": data["u"],
            "v_cm_s": data["v"],
        }
    )
    # Remove empty/ocean-mask entries if any
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["lon", "lat", "u_cm_s", "v_cm_s"])

    u_grid = df.pivot_table(index="lat", columns="lon", values="u_cm_s")
    v_grid = df.pivot_table(index="lat", columns="lon", values="v_cm_s")

    # Align shapes and drop all-NaN rows/cols
    common_lats = u_grid.index.intersection(v_grid.index)
    common_lons = u_grid.columns.intersection(v_grid.columns)
    u_grid = u_grid.loc[common_lats, common_lons].dropna(axis=0, how="all").dropna(axis=1, how="all")
    v_grid = v_grid.loc[u_grid.index, u_grid.columns]

    lon2d, lat2d = np.meshgrid(u_grid.columns.values, u_grid.index.values)
    u = u_grid.values.astype(float) / 100.0  # cm/s -> m/s
    v = v_grid.values.astype(float) / 100.0  # cm/s -> m/s
    speed = np.sqrt(u ** 2 + v ** 2)
    return lon2d, lat2d, u, v, speed


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    tags = ["A", "B", "C", "D"]

    panel_data = []
    for _, when in PHASES:
        data = fetch_uv_map(when)
        lon2d, lat2d, u, v, speed = uv_to_grid(data)
        panel_data.append((lon2d, lat2d, u, v, speed))

    for ax, tag, (label, when), (lon2d, lat2d, u, v, speed) in zip(axes, tags, PHASES, panel_data):
        bm = Basemap(
            projection="cyl",
            llcrnrlon=LON0,
            llcrnrlat=LAT0,
            urcrnrlon=LON1,
            urcrnrlat=LAT1,
            resolution="i",
            ax=ax,
        )
        bm.fillcontinents(color="#efefef", lake_color="#efefef", zorder=1)
        bm.drawcoastlines(linewidth=1.0, color="#2f2f2f", zorder=4)

        # Decimate vectors for legibility
        qx = lon2d[::QUIVER_STRIDE, ::QUIVER_STRIDE]
        qy = lat2d[::QUIVER_STRIDE, ::QUIVER_STRIDE]
        qu = u[::QUIVER_STRIDE, ::QUIVER_STRIDE]
        qv = v[::QUIVER_STRIDE, ::QUIVER_STRIDE]

        bm.quiver(
            qx,
            qy,
            qu,
            qv,
            latlon=True,
            color="#3b44d0",
            scale=35,
            width=0.0019,
            headwidth=2.8,
            headlength=3.2,
            headaxislength=2.9,
            minlength=0.0,
            zorder=5,
        )

        ax.text(0.02, 0.96, tag, transform=ax.transAxes, va="top", ha="left", fontsize=11, fontweight="bold")
        ax.text(0.98, 0.96, when[11:16], transform=ax.transAxes, va="top", ha="right", fontsize=9, color="#2f2f2f")

        ax.set_xticks(np.arange(117, 120.9, 0.5))
        ax.set_yticks(np.arange(23, 25.1, 0.2))
        ax.set_xticks(np.arange(116.8, 120.81, 1.0 / 30.0), minor=True)
        ax.set_yticks(np.arange(23.0, 25.01, 1.0 / 30.0), minor=True)
        ax.tick_params(labelsize=8, direction="in", top=True, right=True)
        ax.tick_params(which="minor", length=1, direction="in", top=True, right=True)
        ax.grid(True, which="major", linestyle="-", linewidth=0.25, color="#d5d5d5", alpha=0.9)
        ax.grid(True, which="minor", linestyle="-", linewidth=0.18, color="#e8e8e8", alpha=0.8)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Add simple per-panel reference text to mimic paper annotations
    axes[0].text(0.23, 0.96, "peak flood\nvelocity", transform=axes[0].transAxes, va="top", ha="left", fontsize=9, color="#2f2f2f")
    axes[1].text(0.24, 0.96, "water slack", transform=axes[1].transAxes, va="top", ha="left", fontsize=9, color="#2f2f2f")
    axes[2].text(0.23, 0.96, "peak ebb\nvelocity", transform=axes[2].transAxes, va="top", ha="left", fontsize=9, color="#2f2f2f")
    axes[3].text(0.24, 0.96, "water slack", transform=axes[3].transAxes, va="top", ha="left", fontsize=9, color="#2f2f2f")
    for ax, txt in zip(axes, ["0.8m/s", "0.3m/s", "0.8m/s", "0.3m/s"]):
        ax.text(0.08, 0.96, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8, color="#2f2f2f")

    fig.subplots_adjust(left=0.055, right=0.995, top=0.995, bottom=0.12, wspace=0.015, hspace=0.015)
    out_png = "examples/case1_figure4_uv.png"
    plt.savefig(out_png, dpi=160)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
