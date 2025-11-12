from __future__ import annotations

import re
import threading
import time
from datetime import date as date_cls, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

import src.config as config
from src.model_utils import get_tide_series, get_tide_time

REFERENCE_TEXT = "TPXO9_atlas_v5 relative to MSL (NTDE 1983–2001)"
USNO_ENDPOINT = "https://aa.usno.navy.mil/api/rstt/oneday"
USNO_CACHE_TTL_SECONDS = 6 * 3600  # 6 hours
TZ_PATTERN = re.compile(r"^([+-]?)(\d{1,2})(?::(\d{2}))?$")

forecast_router = APIRouter(tags=["Tide"])
_usno_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_usno_lock = threading.Lock()


class TideExtreme(BaseModel):
    phen: str
    time: str
    height: float


class AstroEvent(BaseModel):
    phen: str
    time: str


class MoonPhase(BaseModel):
    closestphase: Optional[Dict[str, str]] = None
    curphase: Optional[str] = None
    fracillum: Optional[str] = None


class DayForecast(BaseModel):
    local_date: str
    moonphase: Optional[MoonPhase] = None
    moon: Optional[List[AstroEvent]] = None
    sun: Optional[List[AstroEvent]] = None
    tide: List[TideExtreme]


class TideForecastResponse(BaseModel):
    meta: Dict[str, Any]
    days: List[DayForecast]


@forecast_router.get(
    "/api/tide/forecast",
    response_model=TideForecastResponse,
    summary="Daily tide extremes (high tide/low tide) plus sun/moon phases/events",
)
def tide_forecast(
    lon: float = Query(..., description="Longitude in degrees [-180, 180]"),
    lat: float = Query(..., description="Latitude in degrees [-90, 90]"),
    date: Optional[str] = Query(
        None, description="Local date in YYYY-MM-DD; defaults to today in tz"
    ),
    tz: Union[str, float, int] = Query(
        "+00:00",
        description="Timezone offset, e.g. +08:00, -05:30, 8, -8. Defaults to UTC.",
    ),
):
    """
    Return tide HIGH/LOW events plus sun/moon rise/set and phases for a single local day.

    Sun/moon data courtesy of the U.S. Navy Astronomical Applications Dept. (USNO API: https://aa.usno.navy.mil/data/api).

    #### Usage
    * e.g. `/api/tide/forecast?lon=123.442&lat=25.086&date=2025-01-11&tz=+08:00`
    """
    _validate_coordinates(lon, lat)
    tzinfo, tz_label, tz_request = _parse_timezone(tz)
    local_day = _resolve_local_day(date, tzinfo)
    local_start = datetime(
        local_day.year, local_day.month, local_day.day, tzinfo=tzinfo
    )
    local_end = local_start + timedelta(hours=23, minutes=59, seconds=59)
    start_utc = local_start.astimezone(timezone.utc)
    end_utc = local_end.astimezone(timezone.utc)

    # pad one hour on both ends so extrema at the boundaries still get neighbors
    calc_start = start_utc - timedelta(hours=1)
    calc_end = end_utc + timedelta(hours=1)
    tide_time, tide_datetimes = get_tide_time(calc_start, calc_end)

    mlon = _wrap_longitude(lon)
    z_amp, z_ph = _select_point_constants(mlon, lat)
    tide_series = get_tide_series(
        z_amp,
        z_ph,
        config.cons,
        tide_time,
        format="netcdf",
        unit="cm",
        drop_mask=True,
    )
    tide_events = _build_tide_events(tide_datetimes, tide_series, local_day, tzinfo)

    usno_payload, usno_status = _fetch_usno_oneday(
        local_day.isoformat(), lat, lon, tz_request
    )
    moon_events: Optional[List[AstroEvent]] = None
    sun_events: Optional[List[AstroEvent]] = None
    moon_phase: Optional[MoonPhase] = None

    if usno_payload:
        props = usno_payload.get("properties", {})
        data = props.get("data", {})
        moon_events = _extract_astro_events(
            data.get("moondata"), local_day, tzinfo
        )
        sun_events = _extract_astro_events(data.get("sundata"), local_day, tzinfo)
        moon_phase = _extract_moon_phase(data, tzinfo)

    response = {
        "meta": {
            "lon": float(lon),
            "lat": float(lat),
            "timezone": tz_label,
            "reference": REFERENCE_TEXT,
            "status": usno_status,
        },
        "days": [
            {
                "local_date": local_start.isoformat(),
                "moonphase": moon_phase,
                "moon": moon_events,
                "sun": sun_events,
                "tide": tide_events,
            }
        ],
    }
    return response


def _validate_coordinates(lon: float, lat: float) -> None:
    if lon < -180.0 or lon > 180.0:
        raise HTTPException(
            status_code=400, detail="lon must be between -180 and 180 degrees"
        )
    if lat < -90.0 or lat > 90.0:
        raise HTTPException(
            status_code=400, detail="lat must be between -90 and 90 degrees"
        )


def _parse_timezone(
    tz_value: Union[str, float, int]
) -> Tuple[timezone, str, str]:
    if isinstance(tz_value, (int, float)):
        minutes_total = int(round(float(tz_value) * 60))
    else:
        tz_text = str(tz_value or "").strip()
        if tz_text.upper() == "Z" or tz_text == "":
            minutes_total = 0
        else:
            match = TZ_PATTERN.fullmatch(tz_text)
            if match:
                sign = -1 if match.group(1) == "-" else 1
                hours = int(match.group(2))
                minutes = int(match.group(3) or 0)
                minutes_total = sign * (hours * 60 + minutes)
            else:
                try:
                    minutes_total = int(round(float(tz_text) * 60))
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid tz format. Use ±HH:MM or numeric hours.",
                    ) from exc

    minutes_total = max(-12 * 60, min(14 * 60, minutes_total))
    tzinfo = timezone(timedelta(minutes=minutes_total))
    sign = "+" if minutes_total >= 0 else "-"
    abs_minutes = abs(minutes_total)
    hours = abs_minutes // 60
    minutes = abs_minutes % 60
    label = f"{sign}{hours:02d}:{minutes:02d}"
    hours_float = minutes_total / 60.0
    request_value = (
        str(int(hours_float))
        if minutes_total % 60 == 0
        else f"{hours_float:.4f}".rstrip("0").rstrip(".")
    )
    return tzinfo, label, request_value


def _resolve_local_day(date_str: Optional[str], tzinfo: timezone) -> date_cls:
    if not date_str:
        return datetime.now(tz=tzinfo).date()
    try:
        parsed = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Use YYYY-MM-DD."
        ) from exc
    return parsed.date()


def _wrap_longitude(lon: float) -> float:
    mlon = lon + 360.0 if lon < 0 else lon
    grid_sz = config.gridSz
    if grid_sz is None:
        raise HTTPException(status_code=500, detail="Grid resolution not available")
    zero_nearest = round(0.5 * grid_sz, 3)
    if 0 <= mlon < zero_nearest:
        mlon = zero_nearest
    return mlon


def _select_point_constants(lon: float, lat: float) -> Tuple[np.ndarray, np.ndarray]:
    if config.dz is None or config.cons is None or config.gridSz is None:
        raise HTTPException(
            status_code=500,
            detail="Tide data not initialized. Please retry after startup completes.",
        )
    tol = 0.5 * config.gridSz
    try:
        dsub = config.dz.sel(lon=lon, lat=lat, method="nearest", tolerance=tol)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Requested point is outside the TPXO9 grid coverage.",
        ) from exc

    z_amp = np.squeeze(dsub["z_amp"].values)
    z_ph = np.squeeze(dsub["z_ph"].values)

    if z_amp.ndim != 1 or z_ph.ndim != 1:
        raise HTTPException(
            status_code=500, detail="Unexpected data shape while extracting constants."
        )
    return z_amp, z_ph


def _build_tide_events(
    times: List[datetime], values: np.ndarray, local_day: date_cls, tzinfo: timezone
) -> List[Dict[str, Any]]:
    if len(times) != len(values):
        raise HTTPException(
            status_code=500, detail="Tide series time/value length mismatch."
        )
    events = []
    for entry in _find_local_extrema(times, values):
        local_dt = entry["time"].astimezone(tzinfo)
        if local_dt.date() == local_day:
            time_iso = _isoformat_seconds(local_dt)
            events.append(
                {
                    "phen": entry["type"],
                    "time": time_iso,
                    "height": round(entry["value"], 2),
                }
            )
    return events


def _find_local_extrema(
    times: List[datetime], values: np.ndarray
) -> List[Dict[str, Any]]:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 3:
        return []
    extrema: List[Dict[str, Any]] = []
    for idx in range(1, arr.size - 1):
        prev_v, curr_v, next_v = arr[idx - 1], arr[idx], arr[idx + 1]
        if np.isnan(prev_v) or np.isnan(curr_v) or np.isnan(next_v):
            continue
        kind: Optional[str] = None
        if curr_v > prev_v and curr_v > next_v:
            kind = "HIGH"
        elif curr_v < prev_v and curr_v < next_v:
            kind = "LOW"
        if not kind:
            continue
        refined_time, refined_value = _refine_extremum(
            times[idx - 1], times[idx], times[idx + 1], prev_v, curr_v, next_v
        )
        extrema.append(
            {"type": kind, "time": refined_time, "value": refined_value}
        )
    return extrema


def _refine_extremum(
    prev_time: datetime,
    current_time: datetime,
    next_time: datetime,
    prev_val: float,
    current_val: float,
    next_val: float,
) -> Tuple[datetime, float]:
    denom = prev_val - 2 * current_val + next_val
    if denom == 0:
        return current_time, current_val
    offset_hours = 0.5 * (prev_val - next_val) / denom
    if abs(offset_hours) > 1:
        return current_time, current_val
    refined_time = current_time + timedelta(hours=offset_hours)
    refined_value = current_val - 0.25 * (prev_val - next_val) * offset_hours
    return refined_time, refined_value


def _fetch_usno_oneday(
    date_str: str, lat: float, lon: float, tz_param: str
) -> Tuple[Optional[Dict[str, Any]], str]:
    cache_key = f"{round(lat,4)}:{round(lon,4)}:{date_str}:{tz_param}"
    now = time.time()
    with _usno_lock:
        cached = _usno_cache.get(cache_key)
        if cached and (now - cached[0]) < USNO_CACHE_TTL_SECONDS:
            return cached[1], ""

    params = {"date": date_str, "coords": f"{lat},{lon}", "tz": tz_param}
    try:
        response = requests.get(USNO_ENDPOINT, params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return None, f"USNO status: {exc}"
    except ValueError as exc:
        return None, f"USNO status: invalid JSON ({exc})"

    with _usno_lock:
        _usno_cache[cache_key] = (now, payload)
    return payload, ""


def _extract_astro_events(
    rows: Optional[List[Dict[str, Any]]],
    local_day: date_cls,
    tzinfo: timezone,
) -> Optional[List[AstroEvent]]:
    if not rows:
        return None
    events: List[AstroEvent] = []
    for row in rows:
        iso_time = _assemble_local_iso(local_day, row.get("time"), tzinfo)
        if iso_time:
            events.append(AstroEvent(phen=row.get("phen", ""), time=iso_time))
    return events or None


def _extract_moon_phase(
    data_block: Dict[str, Any], tzinfo: timezone
) -> Optional[MoonPhase]:
    curphase = data_block.get("curphase")
    fracillum = data_block.get("fracillum")
    closest = data_block.get("closestphase")
    cp_out: Optional[Dict[str, str]] = None

    if closest and closest.get("time"):
        cp_time = _assemble_phase_iso(closest, tzinfo)
        if cp_time:
            cp_out = {"time": cp_time, "phase": closest.get("phase")}

    if not any([cp_out, curphase, fracillum]):
        return None
    return MoonPhase(closestphase=cp_out, curphase=curphase, fracillum=fracillum)


def _assemble_local_iso(
    base_date: date_cls, hhmmss: Optional[str], tzinfo: timezone
) -> Optional[str]:
    if not hhmmss:
        return None
    token = hhmmss.strip()
    if not token or not token[0].isdigit():
        return None
    parts = token.split(":")
    if len(parts) < 2:
        return None
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) > 2 else 0
    try:
        local_dt = datetime(
            base_date.year,
            base_date.month,
            base_date.day,
            hour,
            minute,
            second,
            tzinfo=tzinfo,
        )
    except ValueError:
        return None
    return local_dt.isoformat()


def _assemble_phase_iso(
    phase_block: Dict[str, Any], tzinfo: timezone
) -> Optional[str]:
    try:
        year = int(phase_block.get("year"))
        month = int(phase_block.get("month"))
        day = int(phase_block.get("day"))
    except (TypeError, ValueError):
        return None
    hhmm = phase_block.get("time")
    if not hhmm:
        return None
    try:
        phase_date = date_cls(year, month, day)
    except ValueError:
        return None
    return _assemble_local_iso(phase_date, hhmm, tzinfo)


def _isoformat_seconds(dt: datetime) -> str:
    micro = dt.microsecond
    if micro >= 500000:
        dt = dt + timedelta(seconds=1)
    return dt.replace(microsecond=0).isoformat()


__all__ = ["forecast_router"]
