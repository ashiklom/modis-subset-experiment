"""Download the test month locally (fast development loop)."""
import json
import sys
from pathlib import Path

import earthaccess

from modis_subset import cmr

LAT, LON = 41.31, -72.92  # New Haven, CT
START, END = "2024-01-01", "2024-01-31T23:59:59"
OUT = Path("data/local")

cmr.login()
manifest = {}
for sn in sys.argv[1:] or ["MOD35_L2", "MOD03", "MOD04_L2"]:
    gs = cmr.search(sn, LAT, LON, START, END)
    print(sn, len(gs), "granules", round(sum(g.size_mb or 0 for g in gs)), "MB")
    manifest[sn] = [g.__dict__ | {"begin": g.begin.isoformat(), "end": g.end.isoformat(),
                                  "eq_time": None} for g in gs]
    (OUT / sn).mkdir(parents=True, exist_ok=True)
    earthaccess.download([g.url for g in gs], str(OUT / sn), show_progress=False)
Path(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
