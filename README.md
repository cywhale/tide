# Open API to query TPXO9-v5 global tide models

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.10616822.svg)](https://doi.org/10.5281/zenodo.10616822)

#### Swagger API doc

[ODB Tide API manual/online try-out](https://api.odb.ntu.edu.tw/hub/swagger?node=odb_tide_v1)

#### Usage

1. Query tide height and tidal current

* One-point tide height with time-span limitation (<= 30 days, hourly data): e.g. https://eco.odb.ntu.edu.tw/api/tide?lon0=125&lat0=15&start=2023-07-25&end=2023-07-26T01:30:00.000
   
* Get current in bounding-box <= 45x45 in degrees at one time moment(in ISOstring): e.g. https://eco.odb.ntu.edu.tw/api/tide?lon0=125&lon1=135&lat0=15&lat1=30&start=2023-07-25T01:30:00.000
   
2. Get harmonic constituents of TPXO9 model (M2, S2, N2, K2, K1, O1, P1, Q1, Mf, Mm, M4, MS4, MN4, 2N2, S1)

* e.g. https://eco.odb.ntu.edu.tw/api/tide/const?lon=122.36,122.47&lat=25.02,24.82&constituent=k1,m2,n2,o1,p1,s2&complex=amp,ph,hc&append=z,u,v

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

    Ocean Data Bank, National Science and Technology Council, Taiwan. https://doi.org/10.5281/zenodo.7512112. Accessed DAY/MONTH/YEAR from eco.odb.ntu.edu.tw/api/tide. v1.0.
