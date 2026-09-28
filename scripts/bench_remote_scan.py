"""How expensive is it to build HDF4 references remotely (no download)?"""
import time
from modis_subset import hdf4
from modis_subset.io import STATS, Source

BASE = "https://data.laadsdaac.earthdatacloud.nasa.gov/prod-lads"
CASES = [
    ("MOD35_L2/MOD35_L2.A2024001.1510.061.2024002011208.hdf", ["Cloud_Mask", "Latitude", "Longitude"]),
    ("MOD04_L2/MOD04_L2.A2024001.1510.061.2024004222330.hdf", ["AOD_550_Dark_Target_Deep_Blue_Combined", "Latitude", "Longitude"]),
    ("MOD03/MOD03.A2024001.1510.061.2024002005556.hdf", ["Latitude", "Longitude"]),
]
for path, vs in CASES:
    for bs, tail in [(1 << 18, 1 << 20), (1 << 22, 1 << 22)]:
        src = Source(f"{BASE}/{path}")
        src.size()
        STATS.reset()
        t = time.perf_counter()
        refs, st = hdf4.scan(src, vs, block_size=bs, prefetch_tail=tail)
        dt = time.perf_counter() - t
        print(f"{path.split('/')[0]:9s} block={bs>>10}KiB tail={tail>>10}KiB: {dt:6.2f}s  "
              f"{STATS.requests} req  {STATS.bytes/1e6:6.2f} MB of {st['file_size']/1e6:.1f} MB")
    src = Source(f"{BASE}/{path}")
    STATS.reset(); t = time.perf_counter(); src.read_all()
    print(f"{'':9s} full GET: {time.perf_counter()-t:6.2f}s")
