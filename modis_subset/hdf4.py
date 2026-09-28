"""Extract byte-range references for SDS variables in HDF4 (HDF-EOS2) files.

MODIS Collection 6.1 products are HDF4, not HDF5, so the usual
VirtualiZarr/HDF5 machinery does not apply.  kerchunk ships an experimental
HDF4 scanner (``kerchunk.hdf4.HDF4ToZarr``); we reuse its low-level tag
decoders but

* fix its dtype table (floats are big-endian in HDF4; kerchunk decodes them as
  native little-endian, which silently corrupts float data *and* attributes
  such as ``scale_factor``),
* only decode the tags needed for the requested variables (lazy), so that a
  remote scan through :class:`~modis_subset.io.BlockCachedFile` touches as few
  byte blocks as possible,
* keep proper dimension names and full attribute values.

Every MODIS SDS we care about is stored as a *single* zlib-compressed block
(HDF4 "COMPRESSED" special element, not chunked), so a variable reference is
normally just one ``(offset, length)`` pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import kerchunk.hdf4 as kh

from .io import BlockCachedFile, Source

# HDF4 stores numbers big-endian; kerchunk's table forgets this for floats.
kh.dtypes.update({5: ">f4", 6: ">f8"})
_DTYPES = dict(kh.dtypes)


@dataclass
class Chunk:
    index: tuple[int, ...]
    offset: int
    length: int
    compressed: bool


@dataclass
class VarRef:
    name: str
    shape: tuple[int, ...]
    dtype: str
    dims: list[str]
    attrs: dict
    chunk_shape: tuple[int, ...]
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def compressed_bytes(self) -> int:
        return sum(c.length for c in self.chunks)

    def to_json(self):
        return {
            "name": self.name, "shape": list(self.shape), "dtype": self.dtype,
            "dims": self.dims, "attrs": self.attrs, "chunk_shape": list(self.chunk_shape),
            "chunks": [[list(c.index), c.offset, c.length, c.compressed] for c in self.chunks],
        }

    @classmethod
    def from_json(cls, d):
        return cls(d["name"], tuple(d["shape"]), d["dtype"], d["dims"], d["attrs"],
                   tuple(d["chunk_shape"]),
                   [Chunk(tuple(i), o, n, c) for i, o, n, c in d["chunks"]])


def _clean(v):
    if isinstance(v, np.ndarray):
        v = v.tolist()
        return v[0] if len(v) == 1 else v
    if isinstance(v, np.generic):
        return v.item()
    return v


class _Scanner(kh.HDF4ToZarr):
    """kerchunk's HDF4 parser, driven lazily over an arbitrary file object."""

    def __init__(self, fobj):
        super().__init__(path=None)
        self.f = fobj

    def read_dd_list(self):
        assert self.f.read(4) == b"\x0e\x03\x13\x01", "not an HDF4 file"
        self.tags = {}
        while True:
            ddh = self.read_ddh()
            for _ in range(ddh["ndd"]):
                ident, info = self.read_dd()
                self.tags[ident] = info
            if ddh["next"] == 0:
                break
            self.f.seek(ddh["next"])

    def attr(self, ref):
        vh = self._dec("VH", ref)
        if len(vh["names"]) != 1 or vh["names"][0].lower() != "values":
            return None, None
        dt = _DTYPES.get(vh["types"][0])
        vs = self.tags[("VS", ref)]
        self.f.seek(vs["offset"])
        data = self.f.read(vs["length"])
        if dt == "str" or dt is None:
            val = data.split(b"\x00", 1)[0].decode(errors="replace")
        else:
            val = _clean(np.frombuffer(data, np.dtype(dt)).astype(np.dtype(dt).newbyteorder("=")))
        return vh["name"], val

    def sds(self, vg_ref) -> VarRef:
        vg = self._dec("VG", vg_ref)
        out = VarRef(vg["name"], (), "", [], {}, ())
        sd_ref = None
        for t, r in zip(vg["tag"], vg["refs"]):
            if t == "VG":
                child = self._dec("VG", r)
                if child["cls"].startswith(("Dim", "UDim")):
                    out.dims.append(child["name"])
            elif t == "VH":
                k, v = self.attr(r)
                if k:
                    out.attrs[k] = v
            elif t == "NT":
                typ = self._dec("NT", r)["typ"]  # uses the patched table
                out.dtype = "S1" if typ == "str" else typ
            elif t == "SDD":
                out.shape = tuple(self._dec("SDD", r)["dims"])
            elif t == "SD":
                sd_ref = r
        out.chunk_shape = out.shape
        info = self.tags.get(("SD", sd_ref))
        if info is None:
            # SDS declared but never written: no data, all fill value
            out.chunks = []
        elif info["extended"]:
            self.f.seek(info["offset"])
            data = self._dec_extended()
            if isinstance(data, tuple):  # ("DEFLATE", offset, length)
                _, off, n = data
                out.chunks = [Chunk((0,) * len(out.shape), off, n, True)]
            elif isinstance(data, list):  # chunked
                dims = data[-1]
                out.chunk_shape = tuple(d["chunk_length"] for d in dims)
                for c in data[:-1]:
                    idx = tuple(int(i) for i in c[0].split("."))
                    out.chunks.append(Chunk(idx, c[1], c[2], len(c) > 3))
            else:
                raise NotImplementedError(f"unsupported HDF4 special element for {out.name}")
        else:
            out.chunks = [Chunk((0,) * len(out.shape), info["offset"], info["length"], False)]
        # pad dims (dimension vgroups are sometimes missing)
        while len(out.dims) < len(out.shape):
            out.dims.append(f"{out.name}_dim{len(out.dims)}")
        out.dims = out.dims[: len(out.shape)]
        return out


def scan(src: Source | str, variables=None, block_size=1 << 18, prefetch_tail=1 << 20):
    """Return {name: VarRef} for SDS variables in an HDF4 file.

    Parameters
    ----------
    src : Source or path/URL
    variables : iterable of str, optional
        Only resolve these SDS names (default: all).
    block_size, prefetch_tail :
        Tuning for remote scans (see :class:`BlockCachedFile`).

    Returns
    -------
    refs, stats  where stats has the number of byte blocks touched.
    """
    if not isinstance(src, Source):
        src = Source(src)
    f = BlockCachedFile(src, block_size=block_size,
                        prefetch_tail=prefetch_tail if src.remote else 0)
    s = _Scanner(f)
    s.read_dd_list()
    want = set(variables) if variables is not None else None
    out = {}
    for (tag, ref), info in list(s.tags.items()):
        if tag != "VG":
            continue
        vg = s._dec("VG", ref)
        if not vg["cls"].startswith("Var"):
            continue
        if want is not None and vg["name"] not in want:
            continue
        v = s.sds(ref)
        out[v.name] = v
    if want is not None and set(out) != want:
        missing = want - set(out)
        raise KeyError(f"{sorted(missing)} not found in {src.name}")
    return out, {"blocks": f.nblocks_fetched, "block_size": block_size,
                 "file_size": f.size}
