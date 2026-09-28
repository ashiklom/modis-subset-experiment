"""Run every strategy on the local month and check against brute-force truth."""
import time
import pandas as pd
from modis_subset import cmr, timeseries
from modis_subset.io import STATS

LAT, LON = 41.31, -72.92
S, E = "2024-01-01", "2024-01-31T23:59:59"
pairs = cmr.paired("MOD35_L2", LAT, LON, S, E)
truth = pd.read_parquet("data/truth_MOD35_L2.parquet").set_index("key")
kw = dict(local_dir="data/local", refs_root="cache/refs_local", index_root="cache/index_local", pairs=pairs)
for strat in ["brute", "interp5km", "analytic_window", "index_window", "index_window"]:
    STATS.reset(); t = time.perf_counter()
    df = timeseries.extract("MOD35_L2", LAT, LON, S, E, ["Cloud_Mask"], strategy=strat, **kw)
    dt = time.perf_counter() - t
    m = df.set_index("key").join(truth[["row", "col"]], rsuffix="_true")
    m = m[m.in_swath]
    same = ((m.row == m.row_true) & (m.col == m.col_true)).mean()
    print(f"{strat:16s} {dt:6.1f}s  {STATS.bytes/1e6:8.1f} MB  in_swath={len(m)}  exact={same:.3f}  "
          f"locators={df.locator.value_counts().to_dict()}")
df.to_csv("data/MOD35_timeseries_local.csv", index=False)
print(df[["time", "day_night", "row", "col", "dist_km", "cm_cloudiness"]].head(8).to_string())
df4 = timeseries.extract("MOD04_L2", LAT, LON, S, E, ["AOD_550_Dark_Target_Deep_Blue_Combined"],
                         strategy="native", local_dir="data/local", refs_root="cache/refs_local")
print(df4[["time", "row", "col", "dist_km", "in_swath", "AOD_550_Dark_Target_Deep_Blue_Combined"]].head(8).to_string())
