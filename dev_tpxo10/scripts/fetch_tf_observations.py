#!/usr/bin/env python
"""T-F observation FETCH + NORMALIZE step. Produces the sanitized
observations JSON that `tf_observation.py --observations` consumes.

Run in any env with `requests` (production env is fine). This is the
network-touching half of T-F, kept SEPARATE from the scoring harness so
the gate harness itself never fetches.

Sources:
  * NOAA CO-OPS (PRIMARY gate): no token, supports a longer/more stable
    window; water_level, datum=MSL, units=metric (m -> cm), time_zone=GMT
    so timestamps are already UTC.
  * CWA O-B0075-002 (24 h SMOKE): token read from `.env` (CWA_TOKEN) or
    the environment; the token is sent ONLY in the request and is NEVER
    written to the output JSON or any log (asserted before writing).

Output (sanitized — lon/lat/times/heights only, no token, no metadata):
  {"<station>": {"lon": float, "lat": float,
                 "times_utc": ["YYYY-MM-DDTHH:MM:SS", ...],
                 "heights_cm": [float, ...]}, ...}

The normalization functions (`normalize_noaa` / `normalize_cwa`) are pure
and unit-tested with fixtures; the live fetch is a thin requests wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOAA_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
CWA_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-B0075-002"


# ---------- pure normalization (unit-tested) -----------------------------

def normalize_noaa(noaa_json: dict) -> tuple[str, dict] | None:
    """NOAA datagetter JSON (time_zone=GMT, units=metric) -> sanitized
    station record. Heights metres -> cm. Stops at the first missing
    value (NOAA marks gaps with '')."""
    meta = noaa_json.get("metadata")
    if not meta or "data" not in noaa_json:
        return None
    sid = str(meta["id"])
    lon, lat = float(meta["lon"]), float(meta["lat"])
    times, heights = [], []
    for rec in noaa_json["data"]:
        v = rec.get("v")
        if v in ("", None):
            continue
        # NOAA GMT timestamp "YYYY-MM-DD HH:MM" -> ISO UTC
        dt = datetime.strptime(rec["t"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        times.append(dt.strftime("%Y-%m-%dT%H:%M:%S"))
        heights.append(round(float(v) * 100.0, 3))   # m -> cm
    if not times:
        return None
    return sid, {"lon": lon, "lat": lat, "times_utc": times, "heights_cm": heights}


def normalize_cwa(cwa_json: dict, height_unit: str = "m",
                  station_meta: dict | None = None) -> tuple[str, dict] | None:
    """CWA O-B0075-002 JSON -> sanitized station record. TideHeight is in
    METRES in this dataset (height_unit='m' -> *100 to cm). The obs
    response carries only StationID (no coordinates), so lon/lat are taken
    from `station_meta[StationID] = {"lon", "lat"}` (e.g. test/stations_cwa.json)."""
    try:
        loc = cwa_json["Records"]["SeaSurfaceObs"]["Location"][0]
    except (KeyError, IndexError):
        return None
    station = loc["Station"]
    sid = str(station["StationID"])
    meta = (station_meta or {}).get(sid, {})
    lon = meta.get("lon", station.get("StationLongitude") or station.get("Longitude"))
    lat = meta.get("lat", station.get("StationLatitude") or station.get("Latitude"))
    if lon is None or lat is None:
        return None
    lon, lat = float(lon), float(lat)
    scale = 100.0 if height_unit == "m" else 1.0
    times, heights = [], []
    for rec in loc["StationObsTimes"]["StationObsTime"]:
        we = rec.get("WeatherElements", {})
        h = we.get("TideHeight")
        if h in (None, "None", ""):
            continue
        # CWA DateTime is ISO with +08:00 offset -> convert to UTC
        dt = datetime.fromisoformat(rec["DateTime"]).astimezone(timezone.utc)
        times.append(dt.strftime("%Y-%m-%dT%H:%M:%S"))
        heights.append(round(float(h) * scale, 3))
    if not times:
        return None
    return sid, {"lon": lon, "lat": lat, "times_utc": times, "heights_cm": heights}


def assert_token_absent(payload: dict, token: str | None) -> None:
    """Token hygiene: the CWA token must never leak into the output."""
    if token and token in json.dumps(payload):
        raise RuntimeError("ABORT: the CWA token leaked into the output payload")


# ---------- live fetch (thin requests wrappers) --------------------------

def fetch_noaa(station: str, begin: str, end: str, timeout: int = 20):
    import requests
    params = {"product": "water_level", "begin_date": begin, "end_date": end,
              "datum": "MSL", "station": station, "time_zone": "GMT",
              "units": "metric", "format": "json", "interval": "h",
              "application": "ODB_TIDE_TF"}
    r = requests.get(NOAA_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_cwa(station: str, time_from: str, token: str, timeout: int = 20):
    import requests
    params = {"Authorization": token, "StationID": station,
              "WeatherElement": "TideHeight,TideLevel", "sort": "DataTime",
              "timeFrom": time_from}
    r = requests.get(CWA_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _load_cwa_token() -> str | None:
    # prefer an already-set env var; otherwise read .env without echoing it
    tok = os.environ.get("CWA_TOKEN")
    if tok:
        return tok
    envf = REPO / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.strip().startswith("CWA_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["noaa", "cwa"], required=True)
    ap.add_argument("--stations", required=True,
                    help="comma-separated station ids")
    ap.add_argument("--begin", help="NOAA begin date YYYYMMDD")
    ap.add_argument("--end", help="NOAA end date YYYYMMDD")
    ap.add_argument("--hours", type=int, default=24,
                    help="CWA recent window in hours (smoke; default 24)")
    ap.add_argument("--cwa-height-unit", default="m", choices=["cm", "m"])
    ap.add_argument("--station-meta", type=Path,
                    help="JSON {stationID: {lon, lat}} for CWA coords "
                         "(the obs response carries no coordinates)")
    ap.add_argument("--allow-bulk", action="store_true",
                    help="override the conservative CWA station cap")
    ap.add_argument("--out", type=Path, required=True,
                    help="sanitized observations JSON (tracked path for gate evidence)")
    args = ap.parse_args()

    stations = [s.strip() for s in args.stations.split(",") if s.strip()]
    # Conservative CWA guard (free API with transfer limits): refuse a bulk
    # query unless explicitly overridden; the CWA path is a 24h smoke only.
    CWA_STATION_CAP = 5
    if args.source == "cwa" and len(stations) > CWA_STATION_CAP and not args.allow_bulk:
        print(f"REFUSED: CWA is a 24h smoke — {len(stations)} stations exceeds "
              f"the conservative cap of {CWA_STATION_CAP}. Pass --allow-bulk to "
              "override (avoid bulk CWA queries / dense retries).")
        return 2
    out: dict = {}
    token = None
    if args.source == "noaa":
        if not (args.begin and args.end):
            print("NOAA needs --begin and --end (YYYYMMDD)")
            return 2
        for sid in stations:
            try:
                rec = normalize_noaa(fetch_noaa(sid, args.begin, args.end))
            except Exception as e:
                print(f"  NOAA {sid}: fetch/parse failed ({e}) — skipped")
                continue
            if rec:
                out[rec[0]] = rec[1]
                print(f"  NOAA {rec[0]}: {len(rec[1]['times_utc'])} samples")
    else:
        token = _load_cwa_token()
        if not token:
            print("CWA needs CWA_TOKEN in the environment or .env")
            return 2
        smeta = json.loads(args.station_meta.read_text()) if args.station_meta else {}
        time_from = (datetime.now(timezone.utc) - timedelta(hours=args.hours)
                     ).strftime("%Y-%m-%dT%H:%M:%S")
        for sid in stations:
            try:
                rec = normalize_cwa(fetch_cwa(sid, time_from, token),
                                    height_unit=args.cwa_height_unit,
                                    station_meta=smeta)
            except Exception as e:
                print(f"  CWA {sid}: fetch/parse failed — skipped")  # never echo token
                continue
            if rec:
                out[rec[0]] = rec[1]
                print(f"  CWA {rec[0]}: {len(rec[1]['times_utc'])} samples")

    assert_token_absent(out, token)     # token must never be in the payload
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"PASS fetch_tf_observations: {len(out)} stations -> {args.out}\n"
          f"Now run: uv run --project {REPO} python "
          f"dev_tpxo10/scripts/tf_observation.py --observations {args.out} "
          f"--out dev_tpxo10/benchmarks/tf_observation_real_<DATE>.json")
    return 0 if out else 1


if __name__ == "__main__":
    sys.exit(main())
