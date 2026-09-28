"""Read (parts of) HDF4 SDS variables from byte-range references."""

from __future__ import annotations

import zlib
from itertools import product

import numpy as np

from .hdf4 import VarRef
from .io import Source


def _decode(buf: bytes, compressed: bool, dtype, shape):
    if compressed:
        buf = zlib.decompress(buf)
    a = np.frombuffer(buf, dtype=dtype)
    n = int(np.prod(shape))
    if a.size < n:  # edge chunks can be short; pad with zeros
        a = np.concatenate([a, np.zeros(n - a.size, a.dtype)])
    return a[:n].reshape(shape)


def chunk_grid_index(v: VarRef, chunk):
    """Chunk index in units of chunks.

    kerchunk reports the HDF4 chunk-table origin, which is already in chunk
    units.
    """
    return chunk.index


def read(src: Source, v: VarRef, rows: slice | None = None) -> np.ndarray:
    """Read variable ``v`` (native dtype, no scaling).

    ``rows`` restricts the read along the first axis *if* the variable is
    chunked along that axis, only the chunks intersecting ``rows`` are fetched.
    For single-block variables the whole block is fetched and decompressed.
    """
    shape = v.shape
    cshape = v.chunk_shape
    dtype = np.dtype(v.dtype)
    r0, r1 = (0, shape[0]) if rows is None else (rows.start or 0, rows.stop if rows.stop is not None else shape[0])
    r0, r1 = max(0, r0), min(shape[0], r1)
    sel = []
    for c in v.chunks:
        idx = chunk_grid_index(v, c)
        lo = idx[0] * cshape[0]
        if lo < r1 and lo + cshape[0] > r0:
            sel.append((idx, c))
    bufs = src.get_ranges([c.offset for _, c in sel], [c.length for _, c in sel])
    out_shape = (r1 - r0,) + tuple(shape[1:])
    out = np.zeros(out_shape, dtype=dtype.newbyteorder("="))
    for (idx, c), b in zip(sel, bufs):
        blk = _decode(b, c.compressed, dtype, cshape)
        # place block
        dst, srcs = [], []
        for ax, (i, cs, n) in enumerate(zip(idx, cshape, shape)):
            lo, hi = i * cs, min((i + 1) * cs, n)
            if ax == 0:
                lo2, hi2 = max(lo, r0), min(hi, r1)
                dst.append(slice(lo2 - r0, hi2 - r0))
                srcs.append(slice(lo2 - lo, hi2 - lo))
            else:
                dst.append(slice(lo, hi))
                srcs.append(slice(0, hi - lo))
        out[tuple(dst)] = blk[tuple(srcs)]
    return out


def scale(v: VarRef, a: np.ndarray) -> np.ndarray:
    """Apply MODIS scale/offset and mask fill values -> float64 with NaN.

    MODIS atmosphere products use  value = scale_factor * (stored - add_offset).
    """
    a = a.astype("f8")
    fill = v.attrs.get("_FillValue")
    if fill is not None:
        a[a == fill] = np.nan
    vr = v.attrs.get("valid_range")
    if isinstance(vr, list) and len(vr) == 2:
        a[(a < vr[0]) | (a > vr[1])] = np.nan
    sf = v.attrs.get("scale_factor", 1.0)
    off = v.attrs.get("add_offset", 0.0)
    return sf * (a - off)
