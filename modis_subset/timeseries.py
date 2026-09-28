"""Point time series from MODIS level-2 swath products.

    >>> from modis_subset.timeseries import extract
    >>> df = extract("MOD35_L2", 41.31, -72.92, "2024-01-01", "2024-01-31T23:59:59",
    ...              ["Cloud_Mask"], strategy="interp5km")

Strategies for the geolocation step (see :mod:`modis_subset.locate`):

``native``           product's own Latitude/Longitude at the variable's resolution
                     (MOD04_L2, 5 km MOD06/MOD35 fields). No MOD03.
``interp5km``        product's 5 km Latitude/Longitude interpolated to 1 km. No MOD03.
``brute``            full MOD03 Latitude/Longitude.
``analytic_window``  MOD03 scans around the CMR-footprint analytic prediction.
``index_window``     MOD03 scans from the cached spatial index (falls back to
                     ``brute`` and builds the index when it is missing).
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import pandas as pd

from . import cmr, geo, locate, orbit, reader
from .index import IndexCache
from .io import STATS, Source
from .refcache import RefCache

# maximum nearest-pixel distance (km) for the point to count as "in the swath"
MAX_DIST = {1: 5.0, 5: 12.0, 10: 25.0}


def _source(g: cmr.Granule, local_dir) -> Source:
    if local_dir is None:
        return Source(g.url)
    return Source(Path(local_dir) / g.short_name / g.name)


def _spatial_axes(v):
    along = [i for i, d in enumerate(v.dims) if "along" in d.lower()]
    across = [i for i, d in enumerate(v.dims) if "across" in d.lower()]
    if len(along) != 1 or len(across) != 1:
        raise ValueError(f"cannot identify swath dims of {v.name}: {v.dims}")
    return along[0], across[0]


def _indices(v, row, col):
    ia, ic = _spatial_axes(v)
    ranges = [range(n) for n in v.shape]
    ranges[ia], ranges[ic] = [row], [col]
    return list(iproduct(*ranges))


def _read_values(src, v, row, col):
    idx = _indices(v, row, col)
    raw = reader.read_points(src, v, idx)
    extra = [n for i, n in enumerate(v.shape) if i not in _spatial_axes(v)]
    return raw.reshape(extra) if extra else raw[0]


def decode_mod35_byte0(b0: int) -> dict:
    """MOD35 Cloud_Mask byte 0 bit fields."""
    b = int(b0) & 0xFF
    return {
        "cm_determined": b & 1,
        "cm_cloudiness": (b >> 1) & 3,   # 0 cloudy, 1 prob. cloudy, 2 prob. clear, 3 confident clear
        "cm_day": (b >> 3) & 1,
        "cm_sunglint": 1 - ((b >> 4) & 1),
        "cm_snow_ice": 1 - ((b >> 5) & 1),
        "cm_surface": (b >> 6) & 3,      # 0 water, 1 coastal, 2 desert, 3 land
    }


def _one(p: cmr.Granule, g: cmr.Granule | None, variables, lat, lon, strategy,
         local_dir, refs_cache: RefCache, idx_cache: IndexCache):
    t0 = time.perf_counter()
    rec = {"key": p.key, "time": p.begin, "day_night": p.day_night, "granule": p.name}
    psrc = _source(p, local_dir)
    prefs, pinfo = refs_cache.get(psrc, p.short_name)
    pdata = pinfo["data_source"]
    first = prefs[variables[0]]
    ia, ic = _spatial_axes(first)
    grid = (first.shape[ia], first.shape[ic])
    plat, plon = prefs["Latitude"], prefs["Longitude"]
    native = plat.shape == grid
    res = 1 if grid[1] >= 1300 else (5 if grid[1] >= 250 else 10)

    if native or strategy == "native":
        if not native:
            raise ValueError(f"{p.short_name} lat/lon {plat.shape} != data grid {grid}; "
                             "use a 1 km strategy")
        h = locate.brute(pdata, plat, plon, lat, lon)
        used = "native"
    elif strategy == "interp5km":
        h = locate.interp5km(pdata, plat, plon, lat, lon)
        used = strategy
    else:
        if g is None:
            raise ValueError(f"no geolocation granule for {p.name}")
        gsrc = _source(g, local_dir)
        grefs, ginfo = refs_cache.get(gsrc, g.short_name, ["Latitude", "Longitude"])
        gdata = ginfo["data_source"]
        gl, gn = grefs["Latitude"], grefs["Longitude"]
        used = strategy
        if strategy == "index_window" and idx_cache.has(g.name):
            span = idx_cache.scans(g.name, lat, lon)
            if span is None:  # the point is not in this granule's swath
                h = locate.Hit(-1, -1, np.inf)
            else:
                s0, s1 = span
                h = locate.window(gdata, gl, gn, lat, lon, prior_row=(s0 + s1 + 1) * 5,
                                  half_scans=(s1 - s0 + 2) // 2)
        elif strategy == "analytic_window":
            pr, _, _ = orbit.predict(p.polygon, p.day_night, lat, lon, nrows=gl.shape[0])
            h = locate.window(gdata, gl, gn, lat, lon, pr, half_scans=4)
        elif strategy in ("brute", "index_window"):
            la = reader.read(gdata, gl)
            lo = reader.read(gdata, gn)
            i, j, d = geo.nearest(la, lo, lat, lon)
            h = locate.Hit(i, j, d, scans_read=gl.shape[0] // 10)
            if strategy == "index_window":
                idx_cache.put(g.name, la, lo)
                used = "index_window(build)"
        else:
            raise ValueError(strategy)

    rec |= {"row": h.row, "col": h.col, "dist_km": h.dist_km, "locator": used,
            "refs_cached": pinfo["hit"]}
    rec["in_swath"] = bool(h.dist_km <= MAX_DIST[res])
    if rec["in_swath"]:
        for name in variables:
            v = prefs[name]
            vi, vc = _spatial_axes(v)
            r, c = h.row, h.col
            if (v.shape[vi], v.shape[vc]) != grid:  # e.g. 5 km field with 1 km location
                r = min(r * v.shape[vi] // grid[0], v.shape[vi] - 1)
                c = min(c * v.shape[vc] // grid[1], v.shape[vc] - 1)
            val = _read_values(pdata, v, r, c)
            if name == "Cloud_Mask":
                rec["Cloud_Mask_raw"] = [int(x) & 0xFF for x in np.atleast_1d(val)]
                rec |= decode_mod35_byte0(np.atleast_1d(val)[0])
            elif np.ndim(val):
                rec[name] = reader.scale(v, np.atleast_1d(val)).tolist()
            else:
                rec[name] = float(reader.scale(v, np.array([val]))[0])
    rec["seconds"] = time.perf_counter() - t0
    return rec


def extract(short_name, lat, lon, start, end, variables, strategy="interp5km",
            local_dir=None, refs_root="cache/refs", index_root="cache/index",
            workers=8, pairs=None) -> pd.DataFrame:
    """Time series of ``variables`` at (lat, lon) from every granule in range.

    ``local_dir``: read from ``<local_dir>/<short_name>/<file>`` instead of HTTPS.
    ``pairs``: pre-computed ``cmr.paired`` output (skip the CMR query).
    """
    needs_geo = strategy in ("brute", "analytic_window", "index_window")
    if pairs is None:
        if needs_geo:
            pairs = cmr.paired(short_name, lat, lon, start, end)
        else:
            pairs = [(p, None) for p in cmr.search(short_name, lat, lon, start, end)]
    rc, ic = RefCache(refs_root), IndexCache(index_root)
    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(lambda pg: _one(pg[0], pg[1], list(variables), lat, lon,
                                           strategy, local_dir, rc, ic), pairs))
    return pd.DataFrame(recs).sort_values("time").reset_index(drop=True)
