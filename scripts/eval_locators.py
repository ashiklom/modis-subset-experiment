"""Compare locator strategies against brute-force truth (local files)."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from modis_subset import geo, hdf4, locate, orbit, reader
from modis_subset.io import STATS, Source

LAT, LON = 41.31, -72.92
LOCAL = Path("data/local")
truth = pd.read_parquet("data/truth_MOD35_L2.parquet")
rows = []
for _, x in truth.iterrows():
    g = Source(LOCAL / "MOD03" / x.geo)
    p = Source(LOCAL / "MOD35_L2" / x["prod"])
    gref, _ = hdf4.scan(g, ["Latitude", "Longitude"])
    pref, _ = hdf4.scan(p, ["Latitude", "Longitude"])
    la_true = reader.read(g, gref["Latitude"])
    lo_true = reader.read(g, gref["Longitude"])

    def true_dist(i, j):
        return float(np.linalg.norm(geo.ecef(la_true[i, j], lo_true[i, j]) - geo.ecef(LAT, LON)))

    rec = dict(key=x.key, row=x.row, col=x.col, dist=x.dist_km)
    # interp5km
    STATS.reset(); t = time.perf_counter()
    h = locate.interp5km(p, pref["Latitude"], pref["Longitude"], LAT, LON)
    rec |= dict(i5_row=h.row, i5_col=h.col, i5_truedist=true_dist(h.row, h.col),
                i5_bytes=STATS.bytes, i5_s=time.perf_counter() - t)
    # analytic prior + window
    pr, pc, _ = orbit.predict(json.loads(x.polygon), x.day_night, LAT, LON,
                              nrows=gref["Latitude"].shape[0])
    STATS.reset(); t = time.perf_counter()
    h = locate.window(g, gref["Latitude"], gref["Longitude"], LAT, LON, pr, half_scans=4)
    rec |= dict(an_row=h.row, an_col=h.col, an_scans=h.scans_read, an_bytes=STATS.bytes,
                an_s=time.perf_counter() - t)
    # brute via refs
    STATS.reset(); t = time.perf_counter()
    h = locate.brute(g, gref["Latitude"], gref["Longitude"], LAT, LON)
    rec |= dict(bf_bytes=STATS.bytes, bf_s=time.perf_counter() - t)
    rows.append(rec)
df = pd.DataFrame(rows)
df.to_parquet("data/locator_eval.parquet")
ok = df.dist < 2  # point inside the swath
d = df[ok]
print(f"{ok.sum()} granules with the point inside the swath")
print("interp5km exact match:", ((d.i5_row == d.row) & (d.i5_col == d.col)).mean().round(3),
      " |drow|,|dcol| max:", (d.i5_row - d.row).abs().max(), (d.i5_col - d.col).abs().max(),
      " extra distance vs truth (km) max:", (d.i5_truedist - d.dist).max().round(3))
print("analytic+window exact match:", ((d.an_row == d.row) & (d.an_col == d.col)).mean().round(3),
      " scans read median/max:", d.an_scans.median(), d.an_scans.max())
print("MB read  brute(refs) %.2f  window %.3f  interp5km %.2f" % (d.bf_bytes.mean()/1e6, d.an_bytes.mean()/1e6, d.i5_bytes.mean()/1e6))
print("CPU s    brute(refs) %.3f  window %.3f  interp5km %.3f" % (d.bf_s.mean(), d.an_s.mean(), d.i5_s.mean()))
