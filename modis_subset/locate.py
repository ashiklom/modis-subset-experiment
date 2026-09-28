"""Strategies for finding the swath pixel nearest to a point.

All strategies return a :class:`Hit` (row, col, distance) in the *product's*
pixel grid.  1 km products (MOD35_L2, MOD06 1 km fields, ...) need 1 km
geolocation, which lives in MOD03; coarser products (MOD04_L2 10 km, MOD06/
MOD35 5 km fields) carry their own Latitude/Longitude at native resolution.

Strategies
----------
brute            full Latitude/Longitude arrays, exhaustive nearest search.
window           MOD03 scans around a prior (row, col) only; expands until the
                 minimum is interior.  Priors come from:
                   * ``orbit.predict`` (analytic, CMR footprint only), or
                   * a previously solved granule (orbit/track matching), or
                   * a coarse cached index (``index.CoarseIndex``).
interp5km        no MOD03 at all: interpolate the product's embedded 5 km
                 lat/lon to 1 km within each scan, then search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geo, reader
from .hdf4 import VarRef
from .io import Source

ROWS_PER_SCAN = 10


@dataclass
class Hit:
    row: int
    col: int
    dist_km: float
    scans_read: int = 0


def brute(src: Source, lat: VarRef, lon: VarRef, plat, plon) -> Hit:
    la, lo = reader.read_many(src, [lat, lon])
    i, j, d = geo.nearest(la, lo, plat, plon)
    return Hit(i, j, d, scans_read=lat.shape[0] // ROWS_PER_SCAN)


def window(src: Source, lat: VarRef, lon: VarRef, plat, plon, prior_row: float,
           half_scans: int = 1, max_iter: int = 30) -> Hit:
    """Search MOD03 scans ``prior_scan +- half_scans``; grow toward the edge
    where the minimum lands until it is interior (or the granule ends).

    Each iteration is one concurrent batch of range requests (lat + lon chunk
    per new scan).
    """
    nscan = lat.shape[0] // ROWS_PER_SCAN
    k = int(np.clip(round(prior_row / ROWS_PER_SCAN - 0.5), 0, nscan - 1))
    lo_s, hi_s = max(0, k - half_scans), min(nscan, k + half_scans + 1)
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def load(a, b):
        need = [s for s in range(a, b) if s not in cache]
        if not need:
            return
        la, lo_ = reader.read_many(src, [lat, lon], rows=slice(need[0] * 10, (need[-1] + 1) * 10))
        for s in need:
            r = (s - need[0]) * 10
            cache[s] = (la[r:r + 10], lo_[r:r + 10])

    for _ in range(max_iter):
        load(lo_s, hi_s)
        la = np.concatenate([cache[s][0] for s in range(lo_s, hi_s)])
        lo_ = np.concatenate([cache[s][1] for s in range(lo_s, hi_s)])
        i, j, d = geo.nearest(la, lo_, plat, plon, row0=lo_s * 10)
        s = i // 10
        grow_lo = s == lo_s and lo_s > 0
        grow_hi = s == hi_s - 1 and hi_s < nscan
        if not (grow_lo or grow_hi):
            return Hit(i, j, d, scans_read=len(cache))
        # jump: estimate how many scans away the point is from the local
        # along-track pixel spacing, instead of creeping one scan at a time
        step = max(1, half_scans)
        if grow_lo:
            lo_s = max(0, lo_s - step)
        if grow_hi:
            hi_s = min(nscan, hi_s + step)
    return Hit(i, j, d, scans_read=len(cache))


# ---------------------------------------------------------------------------
# 5 km -> 1 km interpolation (no MOD03 needed)
# ---------------------------------------------------------------------------
def interp_5km_to_1km(lat5, lon5, ncols=1354):
    """Interpolate MODIS 5 km geolocation (sampled at 1 km pixel 5i+2, 5j+2)
    to the 1 km grid, *within each scan* (two 5 km rows per 10-line scan:
    detectors 2 and 7), linearly in ECEF.  Crossing scan boundaries is never
    done, which is what makes this work despite the bow-tie overlap.
    """
    nrows = 5 * np.shape(lat5)[0]
    xyz = geo.ecef(np.asarray(lat5, "f8"), np.asarray(lon5, "f8"))  # (406, 270, 3)
    nscan = nrows // 10
    # along-track, per scan: rows at det 2 and 7 -> det 0..9
    x2, x7 = xyz[0::2], xyz[1::2]                       # (203, 270, 3)
    t = (np.arange(10) - 2) / 5.0                       # det -> weight
    along = x2[:, None] + t[None, :, None, None] * (x7 - x2)[:, None]   # (203,10,270,3)
    along = along.reshape(nscan * 10, lat5.shape[1], 3)
    # across-track: cols 5j+2 -> 0..ncols-1 (linear, extrapolate at edges)
    c5 = 5 * np.arange(lat5.shape[1]) + 2
    cols = np.arange(ncols)
    jj = np.clip(np.searchsorted(c5, cols) - 1, 0, len(c5) - 2)
    w = (cols - c5[jj]) / 5.0
    out = along[:, jj] * (1 - w)[None, :, None] + along[:, jj + 1] * w[None, :, None]
    # back to geodetic lat/lon
    x, y, z = out[..., 0], out[..., 1], out[..., 2]
    lon = np.degrees(np.arctan2(y, x))
    p = np.hypot(x, y)
    lat = np.degrees(np.arctan2(z, p * (1 - geo.E2)))   # good enough near surface
    bad = ~(np.abs(np.asarray(lat5)) <= 90)
    if bad.any():
        badrow = np.repeat(bad.any(1), 5)[:nrows]
        lat[badrow] = np.nan
    return lat, lon


def interp5km(src: Source, lat5: VarRef, lon5: VarRef, plat, plon, half_scans=2) -> Hit:
    """Nearest 1 km pixel from the product's own 5 km geolocation.

    Coarse nearest on the 5 km samples first, then 5 km -> 1 km interpolation
    of only the scans around it (+-``half_scans``).
    """
    la5, lo5 = reader.read_many(src, [lat5, lon5])
    i5, _, _ = geo.nearest(la5, lo5, plat, plon)
    nscan = la5.shape[0] // 2
    k = i5 // 2
    s0, s1 = max(0, k - half_scans), min(nscan, k + half_scans + 1)
    la, lo = interp_5km_to_1km(la5[2 * s0:2 * s1], lo5[2 * s0:2 * s1])
    i, j, d = geo.nearest(la, lo, plat, plon, row0=10 * s0)
    return Hit(i, j, d)
