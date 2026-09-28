"""Does the same (row, col) recur for the same orbit track?

Uses MOD04_L2 (10 km, lat/lon embedded -> exact indices with no MOD03) so that a
whole month can be analysed from ~60 MB of files.  Compares a pre-drift year
(2019, Terra on its 16-day WRS-2 repeat) with 2024 (after Terra's 2022-23
orbit lowering).
"""
import sys

import pandas as pd

from modis_subset import timeseries

LAT, LON = 41.31, -72.92
V = "AOD_550_Dark_Target_Deep_Blue_Combined"
for year in sys.argv[1:] or ["2019", "2024"]:
    df = timeseries.extract("MOD04_L2", LAT, LON, f"{year}-01-01", f"{year}-02-28T23:59:59", [V],
                            strategy="native", refs_root="cache/refs_repeat", workers=8)
    df = df[df.dist_km < 10].reset_index(drop=True)
    df["hhmm"] = df.key.str[-4:]
    pairs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            a, b = df.iloc[i], df.iloc[j]
            dd = (b.time - a.time).total_seconds() / 86400
            if abs(int(a.col) - int(b.col)) <= 2:
                pairs.append(dict(a=a.key, b=b.key, days=round(dd, 3), same_hhmm=a.hhmm == b.hhmm,
                                  drow=int(b.row) - int(a.row), dcol=int(b.col) - int(a.col)))
    p = pd.DataFrame(pairs)
    print(f"== {year}: {len(df)} daytime granules, {len(p)} pairs with |dcol|<=2")
    print(p.sort_values("days").to_string(index=False))
