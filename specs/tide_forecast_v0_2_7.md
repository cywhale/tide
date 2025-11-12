# Spec for Codex for revision v0.2.7 of tide_app.py (FastAPI)

## 1) Endpoint

**Path:** `GET /api/tide/forecast`
**Purpose:** For a given point and local date (with a specified timezone), return:

* Tide extremes (HIGH/LOW) and heights for that local day.
* Sun and Moon events from USNO “oneday” (rise/set/transit, civil twilight).
* Moon phase snapshot for that day (current phase, fraction illuminated) and the nearest primary phase.

**Query parameters**

* `lon` (float, required): longitude in degrees, `[-180,180]`.
* `lat` (float, required): latitude in degrees, `[-90,90]`.
* `date` (string, optional): local date, ISO date `YYYY-MM-DD`. Default = “today” in `tz`.
* `tz` (string/number, optional): timezone offset hours from UTC. Accept forms `+08:00`, `-05:30`, `8`, `-8`. Default `+00:00`.

**Time windows**

* Compute the **local** start = `date 00:00:00` and end = `date 23:59:59` in the given `tz`, then convert both to **UTC** for tide calculations.
* Enforce a max window of exactly one local day.

**Response (JSON)**

```json
{
  "meta": {
    "lon": 123,
    "lat": 23,
    "timezone": "+08:00",
    "reference": "TPXO9_atlas_v5 relative to MSL (NTDE 1983–2001)",
    "status": ""
  },
  "days": [
    {
      "local_date": "2025-10-05T00:00:00+08:00",
      "moonphase": {
        "closestphase": {
          "time": "2025-10-07T11:47:00+08:00",
          "phase": "Full Moon"
        },
        "curphase": "Waxing Gibbous",
        "fracillum": "95%"
      },
      "moon": [
        {"phen": "Set",           "time": "2025-10-05T03:25:00+08:00"},
        {"phen": "Rise",          "time": "2025-10-05T16:15:00+08:00"},
        {"phen": "Upper Transit", "time": "2025-10-05T22:17:00+08:00"}
      ],
      "sun": [
        {"phen": "Begin Civil Twilight", "time": "2025-10-05T05:18:00+08:00"},
        {"phen": "Rise",                  "time": "2025-10-05T05:41:00+08:00"},
        {"phen": "Upper Transit",         "time": "2025-10-05T11:36:00+08:00"},
        {"phen": "Set",                   "time": "2025-10-05T17:32:00+08:00"},
        {"phen": "End Civil Twilight",    "time": "2025-10-05T17:54:00+08:00"}
      ],
      "tide": [
        {"phen": "LOW",  "time": "2025-10-05T03:18:00+08:00", "height": 23.4},
        {"phen": "HIGH", "time": "2025-10-05T09:44:00+08:00", "height": 142.1},
        {"phen": "LOW",  "time": "2025-10-05T15:56:00+08:00", "height": 18.7},
        {"phen": "HIGH", "time": "2025-10-05T22:10:00+08:00", "height": 138.2}
      ]
    }
  ]
}
```

Notes:

* All `time` fields are **local** ISO-8601 with offset (e.g., `+08:00`).
* Heights are in **cm**, referenced to **MSL (NTDE 1983–2001)**.
* If a phenomenon doesn’t occur (e.g., Moon continuously above horizon), omit that entry.

## 2) External dependency (USNO)

* Use USNO Astronomical Applications API “Complete Sun and Moon Data for One Day”:

  * `GET https://aa.usno.navy.mil/api/rstt/oneday?date={YYYY-MM-DD}&coords={lat},{lon}&tz={hours}`
  * The `time` values in `moondata`/`sundata` are **local to the provided `tz`**; no conversion needed besides composing them into full ISO strings for the requested `date`. ([美國海軍天文台][1])
  * `closestphase` in this oneday response is also provided there; assemble into local ISO by combining `{year,month,day,time}` with the same `tz`. (Phases service by itself lists UT times; we are **not** calling that service here. Reference only.) ([美國海軍天文台][1])
  * If any errors happened in fetching USNO API, write the error message in response `meta.status`, and keep the output structure:

```
{
  "meta": {
    "lon": 123,
    "lat": 23,
    "timezone": "+08:00",
    "reference": "TPXO9_atlas_v5 relative to MSL (NTDE 1983–2001)",
    "status": "USNO status: 503 server error", #for example. If fetching is ok, just keep an empty "".
  },
  "days": [
    {
      "local_date": "2025-10-05T00:00:00+08:00",
      "moonphase": None,
      "moon": None,
      "sun": None,
      "tide": [
        {"phen": "LOW",  "time": "2025-10-05T03:18:00+08:00", "height": 23.4},
        {"phen": "HIGH", "time": "2025-10-05T09:44:00+08:00", "height": 142.1},
        {"phen": "LOW",  "time": "2025-10-05T15:56:00+08:00", "height": 18.7},
        {"phen": "HIGH", "time": "2025-10-05T22:10:00+08:00", "height": 138.2}
      ]
    }
  ]
}
```

## 3) Tide computation (internal)

* Reuse existing TPXO9 utilities:

  * `get_tide_series(z_amp, z_ph, cons, tide_time, format="netcdf", unit="cm", drop_mask=True)`
  * Data source: your Zarr TPXO9_atlas_v5; select nearest grid cell to (`lon`,`lat`) as in your current `/api/tide`.
* Run `get_tide_series(...)` for variable **z** only → timeseries of sea level (cm).
* Detect local extrema (HIGH/LOW):

  * Use first-difference sign changes to find candidates; optionally refine peak time/height by a quadratic fit on `(i-1, i, i+1)`.
  * Filter resulting events to those whose **local time** falls within the requested local day, then output in local ISO.

## 4) Edge cases & validation

* `lon` wrap: if `lon < 0`, map to `[0,360)` when indexing the Zarr store (as in current code).
* 0° meridian guard: if near zero, apply small epsilon as you do now.
* `tz` parsing:

  * Accept `+/-HH:MM`, `+/-H[.5]`, or integer hours; clamp to plausible range (e.g., `[-12, +14]`).
* Missing sun/moon events: skip the entry (USNO returns `null` when continuous above/below horizon).
* Numerical stability: if quadratic fit falls outside bracket, fall back to the discrete sample.
* Rate limiting: cache USNO responses in-process for `(lat,lon,date,tz)` for a short TTL (e.g., 6h).

## 5) Errors

* `400` on invalid params (`lon/lat` out of range; `date` parse fail; `tz` unparsable).
* `502` if USNO fetch fails after retries; include `source="USNO"` in `error.meta`.
* `500` on unexpected internal errors; log stack and the `(lon,lat,date,tz)` tuple.

## 6) Suggested code skeleton

```python
# tide_app.py (excerpt)
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
import requests, pandas as pd, numpy as np

from .model_utils import get_tide_series  # your existing
from .model_utils import build_tide_time, find_local_extrema  # add if not present
from . import config  # contains gridSz, cons, dz, etc.

router = APIRouter()

class TidePhen(BaseModel):
    phen: str
    time: str       # local ISO8601

class MoonPhase(BaseModel):
    closestphase: dict | None
    curphase: str | None
    fracillum: str | None

class DayBlock(BaseModel):
    local_date: str
    moonphase: MoonPhase | None
    moon: list[TidePhen]
    sun: list[TidePhen]
    tide: list[dict]  # {phen, time, height}

class TideOneDayResponse(BaseModel):
    meta: dict
    days: list[DayBlock]

def parse_tz(tzstr: str) -> timezone:
    try:
        if ":" in tzstr:
            s = 1 if tzstr[0] == "+" else -1
            hh, mm = tzstr[1:].split(":")
            return timezone(s * timedelta(hours=int(hh), minutes=int(mm)))
        else:
            return timezone(timedelta(hours=float(tzstr)))
    except Exception:
        return timezone.utc

def local_iso(dt: datetime, tz: timezone) -> str:
    return dt.astimezone(tz).isoformat()

def assemble_iso(date_str: str, hhmmss: str, tz: timezone) -> str:
    # hhmmss might be "HH:MM" or "HH:MM:SS"
    if len(hhmmss) == 5: hhmmss += ":00"
    base = datetime.fromisoformat(f"{date_str}T{hhmmss}")
    return base.replace(tzinfo=tz).isoformat()

def fetch_usno_oneday(date_str: str, lat: float, lon: float, tz_param: str) -> dict:
    url = "https://aa.usno.navy.mil/api/rstt/oneday"
    r = requests.get(url, params={"date": date_str, "coords": f"{lat},{lon}", "tz": tz_param}, timeout=12)
    r.raise_for_status()
    return r.json()

@router.get("/api/tide/forecast", response_model=TideOneDayResponse, tags=["Tide"])
def tide_forecast(
    lon: float = Query(...),
    lat: float = Query(...),
    date: str | None = Query(None),
    tz: str = Query("+00:00"),
):
    tzinfo = parse_tz(tz)
    local_day = (pd.to_datetime(date).date() if date else datetime.now(tz=tzinfo).date())
    local_start = datetime(local_day.year, local_day.month, local_day.day, 0, 0, 0, tzinfo=tzinfo)
    local_end   = datetime(local_day.year, local_day.month, local_day.day, 23, 59, 59, tzinfo=tzinfo)

    start_utc = local_start.astimezone(timezone.utc)
    end_utc   = local_end.astimezone(timezone.utc)

    # --- tide series (UTC grid) ---
    tide_time, dtime = build_tide_time(start_utc, end_utc)
    mlon = lon + 360 if lon < 0 else lon
    eps = round(0.5*config.gridSz, 3)
    if 0 <= mlon < eps: mlon = eps

    dsub = config.dz.sel(lon=mlon, lat=lat, method="nearest", tolerance=config.gridSz)
    z_amp = dsub["z_amp"].values
    z_ph  = dsub["z_ph"].values
    cm = get_tide_series(z_amp, z_ph, config.cons, tide_time, format="netcdf", unit="cm", drop_mask=True)

    events = find_local_extrema([t.replace(tzinfo=timezone.utc) for t in dtime], cm)

    # Filter to local day and map to local ISO
    tide_list = []
    for ev in events:
        loc = ev["t_refined"].astimezone(tzinfo)
        if loc.date() == local_day:
            tide_list.append({
                "phen": ev["type"], 
                "time": loc.isoformat(),
                "height": round(ev["v_refined"], 2)
            })

    # --- USNO oneday (already local times in tz) ---
    usno = fetch_usno_oneday(local_day.isoformat(), lat, lon, tz)
    pdata = usno.get("properties", {}).get("data", {})
    # Compose moon/sun event ISO strings (same local date)
    moon = []
    for m in pdata.get("moondata", []) or []:
        if m.get("time"): moon.append({"phen": m["phen"], "time": assemble_iso(local_day.isoformat(), m["time"], tzinfo)})
    sun = []
    for s in pdata.get("sundata", []) or []:
        if s.get("time"): sun.append({"phen": s["phen"], "time": assemble_iso(local_day.isoformat(), s["time"], tzinfo)})

    # closestphase: combine Y/M/D + time into local ISO
    cp = pdata.get("closestphase")
    cp_out = None
    if cp and cp.get("time"):
        # Use cp's own date components (may differ from local_day)
        y, m, d = int(cp["year"]), int(cp["month"]), int(cp["day"])
        hhmm = cp["time"] if len(cp["time"]) in (5,8) else f"{cp['time']}:00"
        cp_dt = datetime(y, m, d, int(hhmm[0:2]), int(hhmm[3:5]), int(hhmm[6:8]) if len(hhmm)==8 else 0, tzinfo=tzinfo)
        cp_out = {"time": cp_dt.isoformat(), "phase": cp.get("phase")}

    resp = {
      "meta": {
        "lon": float(lon), "lat": float(lat),
        "timezone": tz,
        "reference": "TPXO9_atlas_v5 relative to MSL (NTDE 1983–2001)"
      },
      "days": [{
        "local_date": local_start.isoformat(),
        "moonphase": {
          "closestphase": cp_out,
          "curphase": pdata.get("curphase"),
          "fracillum": pdata.get("fracillum")
        },
        "moon": moon,
        "sun": sun,
        "tide": tide_list
      }]
    }
    return resp
```

## 7) Tests (high-value)

* `tz=+08:00`: verify UTC window equals previous 16:00Z to same-day 15:59:59Z.
* Compare USNO `sundata.Rise` local time (string) mapped to ISO; should match `oneday` page for a known city/day.
* Ensure tide events fall within the **local** date only.
* Negative longitudes (e.g., `lon=-122.33`) must select correct grid cell after wrapping.
* Edge: Moon doesn’t rise/set → arrays present but certain `time=null` entries are skipped.

---
## Reference

[1]: https://aa.usno.navy.mil/data/api "Application Programming Interface Documentation"
