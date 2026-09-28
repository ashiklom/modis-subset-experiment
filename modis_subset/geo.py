"""Nearest-pixel search on swath lat/lon arrays."""

from __future__ import annotations

import numpy as np

A = 6378.137            # WGS84 semi-major axis, km
E2 = 6.69437999014e-3   # WGS84 first eccentricity squared


def ecef(lat, lon):
    """Geodetic (deg) -> ECEF (km), on the ellipsoid surface."""
    la, lo = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([n * np.cos(la) * np.cos(lo), n * np.cos(la) * np.sin(lo),
                     n * (1 - E2) * np.sin(la)], axis=-1)


def nearest(lat_arr, lon_arr, lat, lon, row0=0):
    """(row, col, distance_km) of the pixel centre nearest to (lat, lon).

    Uses straight-line (chord) distance in ECEF, which is monotone with
    geodesic distance at these scales.  Fill values (|lat| > 90) are ignored.
    ``row0`` is added to the returned row (for windowed searches).
    """
    lat_arr = np.asarray(lat_arr, "f8")
    lon_arr = np.asarray(lon_arr, "f8")
    p = ecef(lat, lon)
    d2 = ((ecef(lat_arr, lon_arr) - p) ** 2).sum(-1)
    d2[~(np.abs(lat_arr) <= 90)] = np.inf
    i, j = np.unravel_index(np.argmin(d2), d2.shape)
    return int(i) + row0, int(j), float(np.sqrt(d2[i, j]))
