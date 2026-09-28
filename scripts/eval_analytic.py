"""Accuracy of the analytic (CMR-footprint) first guess against brute-force truth."""
import json
import pandas as pd
from modis_subset import orbit

LAT, LON = 41.31, -72.92
df = pd.read_parquet("data/truth_MOD35_L2.parquet")
df = df[df.dist_km < 2]
r = []
for _, x in df.iterrows():
    pr, pc, d = orbit.predict(json.loads(x.polygon), x.day_night, LAT, LON)
    r.append(dict(key=x.key, dn=x.day_night, row=x.row, col=x.col, prow=pr, pcol=pc,
                  vz=x.sensor_zenith))
r = pd.DataFrame(r)
r["drow"] = r.prow - r.row
r["dcol"] = r.pcol - r.col
print(r[["drow", "dcol"]].describe().round(2).to_string())
print("abs error quantiles\n", r[["drow", "dcol"]].abs().quantile([.5, .9, 1]).round(1).to_string())
r.to_parquet("data/analytic_eval.parquet")
