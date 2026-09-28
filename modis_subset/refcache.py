"""Persistent cache of HDF4 byte-range references, stored as Parquet.

One Parquet file per granule: ``<root>/<short_name>/<granule name>.parquet``,
one row per chunk::

    url, variable, shape, dtype, dims, chunk_shape, attrs (JSON),
    chunk_index, offset, length, compressed

Variable metadata is repeated on every chunk row of that variable; MOD03's
chunked Latitude/Longitude are 203 rows each, so this is still only a few KB.
Reads go straight to obstore using ``offset``/``length`` (see
:mod:`modis_subset.reader`); there is no icechunk/zarr layer in between,
because icechunk cannot attach an Earthdata Login token to HTTPS reads.

Building references needs the file's HDF4 headers.  Over HTTPS the header
scan is latency-bound (MOD03 metadata is spread over the whole file), so by
default a cold build fetches the whole file once with a single GET and scans it
in memory; the caller can reuse those bytes for the first read.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import hdf4
from .hdf4 import Chunk, VarRef
from .io import STATS, Source

SCHEMA = pa.schema([
    ("url", pa.string()),
    ("variable", pa.string()),
    ("shape", pa.list_(pa.int64())),
    ("dtype", pa.string()),
    ("dims", pa.list_(pa.string())),
    ("chunk_shape", pa.list_(pa.int64())),
    ("attrs", pa.string()),
    ("chunk_index", pa.list_(pa.int64())),
    ("offset", pa.int64()),
    ("length", pa.int64()),
    ("compressed", pa.bool_()),
])


def _json_default(o):
    return str(o)


def to_table(url: str, refs: dict[str, VarRef]) -> pa.Table:
    rows = {k: [] for k in SCHEMA.names}
    for v in refs.values():
        attrs = json.dumps(v.attrs, default=_json_default)
        for c in v.chunks:
            for k, val in [("url", url), ("variable", v.name), ("shape", list(v.shape)),
                           ("dtype", v.dtype), ("dims", v.dims),
                           ("chunk_shape", list(v.chunk_shape)), ("attrs", attrs),
                           ("chunk_index", list(c.index)), ("offset", c.offset),
                           ("length", c.length), ("compressed", c.compressed)]:
                rows[k].append(val)
    return pa.table(rows, schema=SCHEMA)


def from_table(t: pa.Table) -> dict[str, VarRef]:
    out: dict[str, VarRef] = {}
    for r in t.to_pylist():
        v = out.get(r["variable"])
        if v is None:
            v = out[r["variable"]] = VarRef(
                r["variable"], tuple(r["shape"]), r["dtype"], list(r["dims"]),
                json.loads(r["attrs"]), tuple(r["chunk_shape"]))
        v.chunks.append(Chunk(tuple(r["chunk_index"]), r["offset"], r["length"], r["compressed"]))
    return out


class _MemSource(Source):
    """A Source whose bytes are already in memory (cold build: one full GET)."""

    def __init__(self, name: str, data: bytes):
        self.url = name
        self.remote = False
        self._buf = data
        self._size = len(data)

    def get_range(self, start, length):
        return self._buf[start:start + length]

    def get_ranges(self, starts, lengths):
        return [self._buf[s:s + n] for s, n in zip(starts, lengths)]


class RefCache:
    def __init__(self, root="cache/refs"):
        self.root = Path(root)

    def path(self, src: Source, short_name: str) -> Path:
        return self.root / short_name / f"{src.name}.parquet"

    def get(self, src: Source, short_name: str, variables=None, full_get=True):
        """Return ``(refs, info)``.

        ``info["hit"]`` says whether the cache was used.  On a miss the
        references are built (whole-file GET if ``full_get``), written, and
        ``info["data_source"]`` holds an in-memory Source with the file bytes,
        so the caller can do its first read without refetching anything.
        """
        p = self.path(src, short_name)
        if p.exists():
            t = time.perf_counter()
            tbl = pq.read_table(p)
            if variables is not None:
                tbl = tbl.filter(pc.is_in(tbl["variable"], pa.array(list(variables))))
            refs = from_table(tbl)
            if variables is None or set(variables) <= set(refs):
                return refs, {"hit": True, "seconds": time.perf_counter() - t,
                              "data_source": src}
        t = time.perf_counter()
        scan_src = src
        if src.remote and full_get:
            scan_src = _MemSource(src.url, src.read_all())
        # Always scan every SDS so later requests for other variables still hit.
        refs, _ = hdf4.scan(scan_src)
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(to_table(src.url, refs), p, compression="zstd")
        if variables is not None:
            refs = {k: refs[k] for k in variables}
        return refs, {"hit": False, "seconds": time.perf_counter() - t,
                      "data_source": scan_src}
