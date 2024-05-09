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
