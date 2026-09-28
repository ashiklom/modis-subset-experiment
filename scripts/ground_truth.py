"""Brute-force nearest 1 km pixel (full MOD03 lat/lon) for every local granule."""
import json
from pathlib import Path

import pandas as pd

from modis_subset import cmr, geo, hdf4, reader
from modis_subset.io import Source

LAT, LON = 41.31, -72.92
LOCAL = Path("data/local")

pairs = cmr.paired("MOD35_L2", LAT, LON, "2024-01-01", "2024-01-31T23:59:59")
rows = []
for p, g in pairs:
    src = Source(LOCAL / "MOD03" / g.name)
    refs, _ = hdf4.scan(src, ["Latitude", "Longitude", "SensorZenith"])
    la = reader.read(src, refs["Latitude"])
    lo = reader.read(src, refs["Longitude"])
    i, j, d = geo.nearest(la, lo, LAT, LON)
    vz = reader.read(src, refs["SensorZenith"], rows=slice(i, i + 1))[0, j] * 0.01
    rows.append(dict(key=p.key, begin=p.begin, day_night=p.day_night, row=i, col=j,
                     dist_km=d, sensor_zenith=vz, geo=g.name, prod=p.name,
                     polygon=json.dumps(p.polygon)))
    print(p.key, p.day_night, i, j, round(d, 3), round(vz, 1))
df = pd.DataFrame(rows)
df.to_parquet("data/truth_MOD35_L2.parquet")
