"""Remote (HTTPS via obstore) benchmark of all strategies for one month.

Cold = empty reference cache (first contact with each file: one full GET).
Warm = references (Parquet) and, for index_window, the spatial index cached.
"""
import shutil
import sys
import time

import pandas as pd

from modis_subset import cmr, timeseries
from modis_subset.io import STATS

LAT, LON = 41.31, -72.92
S, E = "2024-01-01", "2024-01-31T23:59:59"
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
REFS_A, REFS_B, IDX = "cache/remote/refs_a", "cache/remote/refs_b", "cache/remote/index"
shutil.rmtree("cache/remote", ignore_errors=True)

t = time.perf_counter()
pairs = cmr.paired("MOD35_L2", LAT, LON, S, E)
t_cmr = time.perf_counter() - t
truth = pd.read_parquet("data/truth_MOD35_L2.parquet").set_index("key")

runs = [
    # label, strategy, refs root
    ("cold  brute (+index build)", "index_window", REFS_A),
    ("cold  interp5km (no MOD03)", "interp5km", REFS_B),
    ("warm  brute", "brute", REFS_A),
    ("warm  analytic_window", "analytic_window", REFS_A),
    ("warm  index_window", "index_window", REFS_A),
    ("warm  interp5km", "interp5km", REFS_B),
]
out = []
print(f"CMR search+pairing: {t_cmr:.1f}s for {len(pairs)} granules; workers={WORKERS}")
for label, strat, refs in runs:
    STATS.reset()
    t = time.perf_counter()
    df = timeseries.extract("MOD35_L2", LAT, LON, S, E, ["Cloud_Mask"], strategy=strat,
                            refs_root=refs, index_root=IDX, workers=WORKERS, pairs=pairs)
    wall = time.perf_counter() - t
    m = df.set_index("key").join(truth[["row", "col"]], rsuffix="_t")
    m = m[m.in_swath]
    r = dict(run=label, wall_s=round(wall, 1), MB=round(STATS.bytes / 1e6, 1),
             requests=STATS.requests, MB_per_granule=round(STATS.bytes / 1e6 / len(df), 2),
             exact=round(((m.row == m.row_t) & (m.col == m.col_t)).mean(), 3))
    out.append(r)
    print(r, flush=True)
    df.to_csv(f"data/remote_{strat}_{'warm' if 'warm' in label else 'cold'}.csv", index=False)
res = pd.DataFrame(out)
res.to_csv("data/bench_remote.csv", index=False)
print(res.to_markdown(index=False))
