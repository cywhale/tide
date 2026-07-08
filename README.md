# Open API to query TPXO global tide models

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.10616822.svg)](https://doi.org/10.5281/zenodo.10616822)

#### Swagger API doc

[ODB Tide API manual/online try-out](https://api.odb.ntu.edu.tw/hub/swagger?node=odb_tide_v1)

#### Usage

1. Query tide height and tidal current

* One-point tide height with time-span limitation (<= 30 days, hourly data): e.g. https://eco.odb.ntu.edu.tw/api/tide?lon0=-157.86453&lat0=21.303333&start=2023-07-25&end=2023-07-26
   
* Get tide map in a small bounding box at one time moment (ISO string): e.g. https://eco.odb.ntu.edu.tw/api/tide?lon0=-158.2&lon1=-157.6&lat0=21.0&lat1=21.6&start=2023-07-25T00:00:00&sample=5
   
2. Get harmonic constituents of TPXO model (M2, S2, N2, K2, K1, O1, P1, Q1, Mf, Mm, M4, MS4, MN4, 2N2, S1)

* e.g. https://eco.odb.ntu.edu.tw/api/tide/const?lon=-157.86453,-70.9137&lat=21.303333,41.6212&constituent=m2,k1&complex=amp,ph&append=z,u,v&mode=row

3. Get daily tide extremes with sun/moon information

* e.g. https://eco.odb.ntu.edu.tw/api/tide/forecast?lon=123.442&lat=25.086&date=2026-07-08&tz=+08:00

#### Notes

* API v1.1 uses TPXO10-atlas-v2 by default. TPXO9-atlas-v5 is deprecated.
* TPXO10 reclassifies coastlines at model grid nodes. Some cells that TPXO9 previously served by extrapolation may now return no value.
* `/api/tide` returns z tide height in cm; u and v tidal-current components are cm/s.
* Large bbox map requests are capped after applying `sample`; increase `sample` or shrink the bbox if the API returns a cell-count error.

#### Demo by <a href="https://api.odb.ntu.edu.tw/hub/" target="_blank">Ocean APIverse</a>

[![Clip_for_Tide_API](https://github.com/cywhale/ODB/blob/master/tide/tide_clip01_ogcquery.png)](https://github.com/cywhale/ODB/blob/master/tide/tide_clip01_ogcquery.png)
Demo: [https://twitter.com/bramasolo/status/1754473128078844021](https://twitter.com/bramasolo/status/1754473128078844021)

#### Attribution

* Data source

    Egbert, Gary D., and Svetlana Y. Erofeeva. "Efficient inverse modeling of barotropic ocean tides." Journal of Atmospheric and Oceanic Technology 19.2 (2002): 183-204.
    
* Parts of this API utilize functions provided by pyTMD (https://github.com/tsutterley/pyTMD). We acknowledge and thank the original authors for their contributions.

#### Disclaimer

* The tide model predictions provided by this API are for reference purposes only and are intended to serve as a preliminary resource, not to be considered as definitive for scientific research or risk assessment. Users should understand that no legal liability or responsibility is assumed by the provider of this API for any decisions made based on reliance on this data. Users should conduct their own independent analysis and verification before relying on the data.
  
* 本API提供的模型預測數據僅供參考之用，旨在做為初步的資訊來源，而不應被視為科學研究或風險評估的決定性依據。使用者須理解，對於依賴這些數據所做出的任何決策，本API提供者不承擔任何法律責任或義務。使用者在依賴這些數據前，應進行獨立分析和驗證。

#### Citation

* This API is compiled by [Ocean Data Bank](https://www.odb.ntu.edu.tw) (ODB), and can be cited as:

    Ocean Data Bank, National Science and Technology Council, Taiwan. https://doi.org/10.5281/zenodo.7512112. Accessed DAY/MONTH/YEAR from eco.odb.ntu.edu.tw/api/tide. v1.1.
