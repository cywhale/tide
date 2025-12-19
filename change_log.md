#### ver 0.0.1 create tpxo9.zarr

#### ver 0.0.2 u,v = U/(D/100) from transport (API not yet)/n1

#### ver 0.0.3 u,v = small U/(D/100) problem: use pyTMD extract constant (not yet)/n2

#### ver 0.0.4 small U/(D/100) problem: use shallowLimit or without (not yet)/n3

#### ver 0.0.5 re-fill NA in extract_ATLAS_pytmd both extract_constants and interpolate methods (not yet)/n4

#### ver 0.0.6 first draft-version of tide API, re-fill NA by 3*1 neighbors (not yet)/n5

    -- add /tide, /tide/const to load each harmonic constituent
    -- refine project structure before rerun zarr dataset https://github.com/tsutterley/pyTMD/discussions/241

#### ver 0.0.7 Rescale u,v amplitude by model.scale=1e-4 according to pyTMD/pull/243, fill NA/n7

    -- small fix TypeError: expected unicode string, found 15 problem in constituents, fill NA/n11

#### ver 0.0.8 Add nearest method to locate points by Zarr; Validate api results with Jen's paper

    -- change to use pipenv for package version management

#### ver 0.0.9 Use vectorized/nearest method to solve mult-point constituents/fix datetime, mode None problems

    -- change wide dataframe format is default. Specify mode=long if long format is preferred
    -- /tide/const example to plot tidal ellipses. Parallel re-fill-NA savings
    -- fix Parallel re-fill-NA savings deadlock problem by Dask version
    -- add FastAPI pydantic response_model
    -- fix bounding box convert to (0,360) bug
    -- fill 2x2 neighbor NA. Note that trouble to mix tpxo9.zarr/_fillna.zarr should be avoided
    -- small package upgrade/fill_NA trials(15th --> 18th)

#### ver 0.1.0 Add API Readme.md, disclaimer, fastapi lifespan for package upgrade

    -- various package upgrade/test(18->25th), dev files small update

#### ver 0.1.1 Let constituents oscillation contribute 0 not mask as NA

    -- small update in dev files for testing/README.md move swagger doc to Ocean APIverse
    -- various package upgrade/n2/test(->35th), dev files small update

#### ver 0.1.2 Start validation with observation(CWA)/model(Marea API) data

    -- Add validation with NOAA tide observation data api.tidesandcurrents.noaa.gov/
    -- Modify testbench (for CWA, NOAA) bugs e.g. local time (NOAA) to UTC
    -- small update testbench, tpxo9.zarr(->36th), dev files

#### ver 0.2.0 Fix get_tide_series u,v unit conversion bug, upgrade pyTMD 2.1.4 breaking change

    -- a temporary solution to make xarry internal index error when lon=0 but actually at -4.06e-6 (floating error)

#### ver 0.2.1 Fix shared dask worker leak by unique naming scheme for Dask tasks/major package upgrade (numpy v2)
#### ver 0.2.2 Compare Tide API with CMEMS cmems_mod_glo_phy_anfc_merged-uv_PT1H-i

    -- pacakge upgrade/testing files update (->47th)
    -- refine the plots that compare Tide API with CMEMS

#### ver 0.2.3 Fix /api/tide/const one-point not consistent with multi-pts/defaultly use nearest method

    -- fix /api/tide one-point use nearest method error/add error handling when given consitituents but too less

#### ver 0.2.4 End of filling NA of tpxo9.zarr
#### ver 0.2.5 Add a 'truncate' mode: lon/lat/tide to 5/5/3 decimal places. Fix tide-height unit in cm in 'map'. 

    -- fix introduced bug due to previous modification: tide height at time_series has wrong unit scaling and without truncate mode

#### ver 0.2.6 Improve Swagger doc (fix numcodecs==0.15.1 issue zarr-developers/numcodecs/issues/721)

#### ver 0.2.7 Add `/api/tide/forecast` daily summary endpoint and contributor guide

    -- Extracted forecast logic into `src/tide_forecast.py`, calling TPXO9 series + USNO oneday data with caching/robust TZ parsing
    -- Standardized tide timestamps to second precision and surfaced external API failures via `meta.status`
    -- Added AGENTS.md contributor guide and documented the new route wiring in FastAPI; requirements.txt already refreshed for package upgrades

#### ver 0.2.8 Decouple shared Dask service and add reconnect-capable client
