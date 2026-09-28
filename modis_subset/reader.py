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


def read_many(src: Source, vs, rows: slice | None = None) -> list[np.ndarray]:
    """Like :func:`read` for several variables, with one batched request."""
    plans = [_plan(v, rows) for v in vs]
    flat = [c for _, sel, _ in plans for _, c in sel]
    bufs = iter(src.get_ranges([c.offset for c in flat], [c.length for c in flat]))
    out = []
    for v, (rr, sel, _) in zip(vs, plans):
        out.append(_assemble(v, rr, sel, [next(bufs) for _ in sel]))
    return out


def _plan(v: VarRef, rows):
    shape, cshape = v.shape, v.chunk_shape
    r0, r1 = (0, shape[0]) if rows is None else (rows.start or 0, rows.stop if rows.stop is not None else shape[0])
    r0, r1 = max(0, r0), min(shape[0], r1)
    sel = []
    for c in v.chunks:
        idx = chunk_grid_index(v, c)
        lo = idx[0] * cshape[0]
        if lo < r1 and lo + cshape[0] > r0:
            sel.append((idx, c))
    return (r0, r1), sel, None


def _assemble(v: VarRef, rr, sel, bufs):
    shape, cshape = v.shape, v.chunk_shape
    dtype = np.dtype(v.dtype)
    r0, r1 = rr
    out_shape = (r1 - r0,) + tuple(shape[1:])
    fill = v.attrs.get("_FillValue", 0)
    out = np.full(out_shape, fill if np.isscalar(fill) else 0, dtype=dtype.newbyteorder("="))
    for (idx, c), b in zip(sel, bufs):
        blk = _decode(b, c.compressed, dtype, cshape)
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


def read(src: Source, v: VarRef, rows: slice | None = None) -> np.ndarray:
    """Read variable ``v`` (native dtype, no scaling).

    ``rows`` restricts the read along the first axis: if the variable is
    chunked along that axis, only the chunks intersecting ``rows`` are fetched.
    For single-block variables the whole block is fetched and decompressed.
    """
    return read_many(src, [v], rows)[0]


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


def read_points(src: Source, v: VarRef, indices, first_fetch_margin=1.15,
                min_fetch=1 << 16) -> np.ndarray:
    """Values of ``v`` at a list of full index tuples, touching as little of
    the compressed stream as possible.

    zlib is a stream, so element ``e`` of a single-block variable only needs
    the compressed prefix that decodes to ``(e + 1) * itemsize`` bytes.  We
    guess that prefix from the block's average compression ratio, fetch it,
    and fetch more (doubling) only if it was not enough.  For a MOD35
    ``Cloud_Mask[0, row, col]`` this is on average ~1/12 of the block.

    Chunked variables fall back to reading only the chunk(s) involved.
    """
    idx = [tuple(int(i) for i in t) for t in indices]
    dtype = np.dtype(v.dtype)
    if len(v.chunks) != 1 or not v.chunks[0].compressed:
        out = []
        for t in idx:
            a = read(src, v, rows=slice(t[0], t[0] + 1))
            out.append(a[(0,) + t[1:]])
        return np.array(out, dtype=dtype.newbyteorder("="))
    c = v.chunks[0]
    flat = np.ravel_multi_index(np.array(idx).T, v.shape)
    need = int((flat.max() + 1) * dtype.itemsize)
    total = int(np.prod(v.shape)) * dtype.itemsize
    guess = int(need / total * c.length * first_fetch_margin) + min_fetch
    d = zlib.decompressobj()
    got, pos, fetch = [], 0, min(c.length, guess)
    have = 0
    while have < need:
        if pos >= c.length:
            raise ValueError("compressed stream ended early")
        n = min(fetch, c.length - pos)
        part = src.get_range(c.offset + pos, n)
        pos += n
        chunk = d.decompress(d.unconsumed_tail + part, need - have)
        got.append(chunk)
        have += len(chunk)
        fetch = max(fetch, min_fetch) * 2
    buf = b"".join(got)[:need]
    a = np.frombuffer(buf, dtype=dtype)
    return a[flat].astype(dtype.newbyteorder("="))


def point_fraction_read(v: VarRef, indices) -> float:
    """Rough fraction of the compressed block needed for ``read_points``."""
    flat = np.ravel_multi_index(np.array([tuple(t) for t in indices]).T, v.shape)
    return float((flat.max() + 1) / np.prod(v.shape))
