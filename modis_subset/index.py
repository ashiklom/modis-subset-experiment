"""Cached coarse spatial index: which scans of a granule cover a location.

Built once per geolocation granule, at the moment the full lat/lon arrays are
in memory anyway (the cold reference build downloads the whole MOD03 file).
For every pixel we compute a spatial bin id; the index stores, per bin, the
range of scans whose pixels fall in it.  A later point query is then:

    bin(point) -> [scan_min, scan_max] -> read those 1-3 MOD03 scans -> exact search

Bins are S2-style cells: the sphere is projected on the six cube faces and
each face is split into 2**LEVEL x 2**LEVEL equal-angle cells (S2's "tan"
projection, without the Hilbert-curve ordering, which a point lookup does not
need).  This is fully vectorised in numpy; the pure-Python ``s2sphere`` takes
seconds per granule for 2.7 M pixels.  LEVEL=11 gives ~5 km cells.  The 3x3
neighbourhood of the query cell is checked so that a point near a cell edge
still finds the scans holding its nearest pixel.

Stored as Parquet: ``<root>/<geo granule>.parquet`` with columns
``cell (uint64), scan_min (int16), scan_max (int16)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LEVEL = 11            # 2**11 cells per face edge -> ~ 90deg/2048 ~ 4.9 km
ROWS_PER_SCAN = 10


def _xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def cell_ids(lat, lon, level=LEVEL):
    """Vectorised cube-face cell ids (face, i, j) packed into uint64."""
    p = _xyz(np.asarray(lat, "f8"), np.asarray(lon, "f8"))
    ax = np.argmax(np.abs(p), -1)
    sign = np.take_along_axis(p, ax[..., None], -1)[..., 0] > 0
    face = ax + 3 * (~sign)
    # the two remaining coordinates, projected on the face
    u_ax = (ax + 1) % 3
    v_ax = (ax + 2) % 3
    w = np.abs(np.take_along_axis(p, ax[..., None], -1)[..., 0])
    u = np.take_along_axis(p, u_ax[..., None], -1)[..., 0] / w
    v = np.take_along_axis(p, v_ax[..., None], -1)[..., 0] / w
    # equal-angle transform (like S2's tan projection) -> [0, 1)
    n = 1 << level
    s = np.clip(((np.arctan(u) / (np.pi / 4)) + 1) / 2, 0, 1 - 1e-12)
    t = np.clip(((np.arctan(v) / (np.pi / 4)) + 1) / 2, 0, 1 - 1e-12)
    i = (s * n).astype(np.uint64)
    j = (t * n).astype(np.uint64)
    return (face.astype(np.uint64) << np.uint64(58)) | (i << np.uint64(29)) | j


def neighbours(cell):
    cell = np.uint64(cell)
    face = cell >> np.uint64(58)
    i = (cell >> np.uint64(29)) & np.uint64((1 << 29) - 1)
    j = cell & np.uint64((1 << 29) - 1)
    out = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii, jj = int(i) + di, int(j) + dj
            if ii < 0 or jj < 0:
                continue  # face edges: good enough for mid-latitude points
            out.append((int(face) << 58) | (ii << 29) | jj)
    return np.array(out, dtype=np.uint64)


def build(lat, lon, level=LEVEL) -> pa.Table:
    lat = np.asarray(lat)
    ok = np.abs(lat) <= 90
    c = cell_ids(np.where(ok, lat, 0), np.where(ok, lon, 0), level)
    scan = np.broadcast_to((np.arange(lat.shape[0]) // ROWS_PER_SCAN)[:, None], lat.shape)
    c, scan = c[ok], scan[ok]
    order = np.argsort(c, kind="stable")
    c, scan = c[order], scan[order]
    uniq, start = np.unique(c, return_index=True)
    smin = np.minimum.reduceat(scan, start)
    smax = np.maximum.reduceat(scan, start)
    return pa.table({"cell": pa.array(uniq, pa.uint64()),
                     "scan_min": pa.array(smin.astype(np.int16)),
                     "scan_max": pa.array(smax.astype(np.int16))})


class IndexCache:
    def __init__(self, root="cache/index"):
        self.root = Path(root)

    def path(self, geo_name):
        return self.root / f"{geo_name}.parquet"

    def has(self, geo_name):
        return self.path(geo_name).exists()

    def put(self, geo_name, lat, lon):
        p = self.path(geo_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(build(lat, lon), p, compression="zstd")

    def scans(self, geo_name, lat, lon):
        """(scan_min, scan_max) that may contain the nearest pixel, or None."""
        t = pq.read_table(self.path(geo_name))
        cells = t["cell"].to_numpy()
        m = np.isin(cells, neighbours(cell_ids(lat, lon)))
        if not m.any():
            return None
        return int(t["scan_min"].to_numpy()[m].min()), int(t["scan_max"].to_numpy()[m].max())
