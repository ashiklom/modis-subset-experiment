"""Validate our HDF4 byte-range references against pyhdf for every SDS."""
import sys
import numpy as np
from pyhdf.SD import SD
from modis_subset import hdf4, reader
from modis_subset.io import Source

for fn in sys.argv[1:]:
    refs, st = hdf4.scan(fn)
    sd = SD(fn)
    bad = 0
    for k, v in refs.items():
        a = reader.read(Source(fn), v)
        t = sd.select(k)[:]
        if not np.array_equal(a, t):
            bad += 1
            print("  DATA MISMATCH", k)
        for ak, av in sd.select(k).attributes().items():
            mine = v.attrs.get(ak)
            if isinstance(av, list) and len(av) == 1:
                av = av[0]
            if isinstance(av, str):
                av = av.rstrip("\x00")
            ok = mine == av or (isinstance(av, float) and np.isclose(mine, av)) or \
                (isinstance(av, list) and np.allclose(mine, av))
            if not ok:
                bad += 1
                print("  ATTR MISMATCH", k, ak, repr(mine)[:60], repr(av)[:60])
    nchunk = {k: len(v.chunks) for k, v in refs.items() if len(v.chunks) > 1}
    print(fn.rsplit("/", 1)[-1], f"{len(refs)} SDS, mismatches={bad}, chunked vars={nchunk}")
