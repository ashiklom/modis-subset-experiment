"""Analytic first guess of the MODIS 1 km (row, col) for a point.

Uses only CMR metadata that the granule search already returns: the granule
footprint polygon (four corners = ends of the first and last scan lines).

* Along-track: the nadir track is approximated by the great circle through the
  midpoints of the first and last scan lines; the fractional position of the
  point's projection on it gives the row.
* Across-track: the signed angular distance ``gamma`` of the point from that
  great circle is converted to a MODIS scan angle with plane trigonometry of a
  spherical Earth, ``theta = atan(R sin(gamma) / (R + h - R cos(gamma)))``,
  and to a sample index with the fixed 1 km angular step.

The corner order of the CMR polygon tells us which corners are row 0 / col 0
(see :func:`scan_lines`).
"""

from __future__ import annotations

import numpy as np

R = 6371.0            # km, mean Earth radius
H = 705.0             # km, nominal Terra/Aqua altitude
NSAMP = 1354
NROWS = 2030
DTHETA = np.radians(110.0) / NSAMP   # rad per 1 km sample (+-55 deg scan)


def _xyz(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _unit(v):
    return v / np.linalg.norm(v)


def _mid(a, b):
    return _unit(a + b)


def scan_lines(polygon, day_night=None):
    """Return ((a0, b0), (a1, b1)): endpoints of the first and last scan line.

    ``a`` is sample 0 and ``b`` sample 1353.  LAADS writes the MODIS GRing to
    CMR in a fixed order, verified against MOD03 corner pixels:
    (last row, col 0), (last row, col 1353), (first row, col 1353),
    (first row, col 0).
    """
    pts = [_xyz(*p) for p in polygon]
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) != 4:
        raise ValueError("expected a 4-corner footprint")
    p0, p1, p2, p3 = pts
    return (p3, p2), (p0, p1)


def predict(polygon, day_night, lat, lon, h=H, nrows=NROWS):
    """Predicted (row, col) as floats, plus diagnostics."""
    (a0, b0), (a1, b1) = scan_lines(polygon, day_night)
    n0, n1 = _mid(a0, b0), _mid(a1, b1)
    pole = _unit(np.cross(n0, n1))            # normal of the nadir great circle
    p = _xyz(lon, lat)
    gamma = np.arcsin(np.clip(np.dot(p, pole), -1, 1))    # signed cross-track angle
    proj = _unit(p - np.dot(p, pole) * pole)
    total = np.arctan2(np.dot(np.cross(n0, n1), pole), np.dot(n0, n1))
    along = np.arctan2(np.dot(np.cross(n0, proj), pole), np.dot(n0, proj))
    row = along / total * nrows - 0.5
    theta = np.arctan2(R * np.sin(gamma), R + h - R * np.cos(gamma))
    # sample 0 is on the side of endpoint "a" or "b"; decide by the sign of a0
    side_a = np.sign(np.dot(a0, pole))
    col = (NSAMP - 1) / 2 - side_a * theta / DTHETA
    return float(row), float(col), {"gamma_km": float(gamma * R), "theta_deg": float(np.degrees(theta))}
