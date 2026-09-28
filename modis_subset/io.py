"""Authentication (earthaccess) and byte-range I/O (obstore) helpers.

Everything that touches bytes goes through a :class:`Source`, which is either a
local file or a remote HTTPS object read with obstore.  Both expose the same
``get_range`` / ``get_ranges`` API so the rest of the code does not care where
the bytes live.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import obstore
from obstore.store import HTTPStore, LocalStore


# --------------------------------------------------------------------------
# Earthdata login
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def login():
    """Log in with earthaccess using EARTHDATA_LOGIN_USER / EARTHDATA_LOGIN_PASSWORD.

    earthaccess itself looks for EARTHDATA_USERNAME / EARTHDATA_PASSWORD, so map
    the variable names used in this environment onto those.
    """
    import earthaccess

    for src, dst in [
        ("EARTHDATA_LOGIN_USER", "EARTHDATA_USERNAME"),
        ("EARTHDATA_LOGIN_PASSWORD", "EARTHDATA_PASSWORD"),
    ]:
        if src in os.environ and dst not in os.environ:
            os.environ[dst] = os.environ[src]
    auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        raise RuntimeError("earthaccess login failed")
    return auth


def edl_token() -> str:
    return login().token["access_token"]


@lru_cache(maxsize=None)
def http_store(base_url: str) -> HTTPStore:
    """obstore HTTPStore for an https host, sending the EDL bearer token.

    reqwest (under obstore) drops the Authorization header on the cross-origin
    redirect to the signed CloudFront/S3 URL, which is what we want.
    """
    return HTTPStore.from_url(
        base_url,
        client_options={
            "default_headers": {"Authorization": f"Bearer {edl_token()}"},
            "timeout": "300s",
        },
    )


# --------------------------------------------------------------------------
# Signed-URL cache
# --------------------------------------------------------------------------
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


class SignedURLCache:
    """Remember the pre-signed CloudFront URL that LAADS redirects to.

    Every request to ``data.laadsdaac.earthdatacloud.nasa.gov`` answers with a
    303 to a signed CloudFront URL (valid ~1 h, see its ``Expires``), so each
    range read costs two round trips.  Resolving the redirect once per file and
    pointing an obstore HTTPStore at the signed URL removes one of them.  The
    redirect is resolved with urllib (obstore cannot report a redirect target);
    all data bytes still go through obstore.  Persisted to a small JSON file so
    that it also helps across processes.
    """

    def __init__(self, path="cache/signed_urls.json", margin_s=300):
        self.path = Path(path)
        self.margin = margin_s
        self.lock = threading.Lock()
        try:
            self.urls = json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError):
            self.urls = {}

    def _valid(self, signed):
        exp = parse_qs(urlsplit(signed).query).get("Expires")
        return bool(exp) and int(exp[0]) > time.time() + self.margin

    def get(self, url):
        with self.lock:
            signed = self.urls.get(url)
        if signed and self._valid(signed):
            return signed
        t = time.perf_counter()
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {edl_token()}", "Range": "bytes=0-0"})
        try:
            urllib.request.build_opener(_NoRedirect).open(req, timeout=60).close()
            signed = None  # no redirect: serve directly
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308):
                raise
            signed = e.headers["Location"]
        STATS.add(1, 0, time.perf_counter() - t)
        with self.lock:
            self.urls[url] = signed
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.urls))
        return signed


SIGNED = None  # set to a SignedURLCache() to enable (see enable_signed_urls)


def enable_signed_urls(path="cache/signed_urls.json"):
    global SIGNED
    SIGNED = SignedURLCache(path)


@lru_cache(maxsize=4096)
def _signed_store(signed_url: str) -> HTTPStore:
    return HTTPStore.from_url(signed_url, client_options={"timeout": "300s"})


# --------------------------------------------------------------------------
# Byte sources
# --------------------------------------------------------------------------
@dataclass
class IOStats:
    requests: int = 0
    bytes: int = 0
    seconds: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, nreq, nbytes, secs):
        with self.lock:
            self.requests += nreq
            self.bytes += nbytes
            self.seconds += secs

    def reset(self):
        with self.lock:
            self.requests = self.bytes = 0
            self.seconds = 0.0

    def as_dict(self):
        return {"requests": self.requests, "bytes": self.bytes, "io_seconds": round(self.seconds, 3)}


STATS = IOStats()


class Source:
    """A byte-addressable file, local or remote."""

    def __init__(self, url_or_path: str):
        self.url = str(url_or_path)
        if self.url.startswith(("http://", "https://")):
            parts = urlsplit(self.url)
            # Use the first path component (e.g. /prod-lads) as the store root
            # so that one HTTPStore (one connection pool) is shared per bucket.
            first, _, rest = parts.path.lstrip("/").partition("/")
            self.store = http_store(f"{parts.scheme}://{parts.netloc}/{first}")
            self.path = rest
            self.remote = True
            if SIGNED is not None:
                signed = SIGNED.get(self.url)
                if signed:
                    self.store, self.path = _signed_store(signed), ""
        else:
            p = Path(self.url).resolve()
            self.store = _local_store(str(p.parent))
            self.path = p.name
            self.remote = False
        self._size = None

    def __repr__(self):
        return f"Source({self.url!r})"

    @property
    def name(self) -> str:
        return self.url.rsplit("/", 1)[-1]

    def size(self) -> int:
        if self._size is None:
            self._size = obstore.head(self.store, self.path)["size"]
        return self._size

    def get_range(self, start: int, length: int) -> bytes:
        t = time.perf_counter()
        b = bytes(obstore.get_range(self.store, self.path, start=start, length=length))
        STATS.add(1, len(b), time.perf_counter() - t)
        return b

    def get_ranges(self, starts, lengths) -> list[bytes]:
        if not len(starts):
            return []
        t = time.perf_counter()
        # obstore coalesces nearby ranges and issues requests concurrently
        bs = obstore.get_ranges(self.store, self.path, starts=list(starts), lengths=list(lengths))
        out = [bytes(b) for b in bs]
        STATS.add(len(out) if self.remote else 1, sum(map(len, out)), time.perf_counter() - t)
        return out

    def read_all(self) -> bytes:
        t = time.perf_counter()
        b = bytes(obstore.get(self.store, self.path).bytes())
        STATS.add(1, len(b), time.perf_counter() - t)
        return b


@lru_cache(maxsize=None)
def _local_store(root: str) -> LocalStore:
    return LocalStore(root)


class BlockCachedFile:
    """Minimal read/seek/tell file object over a :class:`Source`.

    Reads are served from fixed-size blocks that are fetched lazily (one range
    request per missing run of blocks).  The HDF4 header scanner does tens of
    thousands of tiny reads, so this turns them into a handful of requests.
    An optional ``prefetch_tail`` grabs the end of the file up front, which is
    where MODIS L2 products keep most of their HDF4 metadata.
    """

    def __init__(self, src: Source, block_size: int = 1 << 18, prefetch_tail: int = 0,
                 prefetch_head: int = 0):
        self.src = src
        self.bs = block_size
        self.blocks: dict[int, bytes] = {}
        self.pos = 0
        self.size = src.size()
        nblk = (self.size + self.bs - 1) // self.bs
        want = set()
        if prefetch_head:
            want |= set(range(0, min(nblk, (prefetch_head + self.bs - 1) // self.bs)))
        if prefetch_tail:
            first = max(0, (self.size - prefetch_tail) // self.bs)
            want |= set(range(first, nblk))
        self._fetch(sorted(want))

    def _fetch(self, idx):
        idx = [i for i in idx if i not in self.blocks]
        if not idx:
            return
        starts = [i * self.bs for i in idx]
        lengths = [min(self.bs, self.size - s) for s in starts]
        for i, b in zip(idx, self.src.get_ranges(starts, lengths)):
            self.blocks[i] = b

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        b0, b1 = self.pos // self.bs, (self.pos + n - 1) // self.bs
        self._fetch(range(b0, b1 + 1))
        buf = b"".join(self.blocks[i] for i in range(b0, b1 + 1))
        s = self.pos - b0 * self.bs
        self.pos += n
        return buf[s : s + n]

    @property
    def nblocks_fetched(self):
        return len(self.blocks)
