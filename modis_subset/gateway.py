"""Tiny local S3-compatible *read-only* gateway in front of LAADS HTTPS.

Why this exists
---------------
Icechunk can read virtual chunks from S3/GCS/Azure/HTTP containers, but its
HTTP container cannot attach an Earthdata Login bearer token (object_store's
HTTP client config has no "default headers" key), so every virtual chunk read
against ``https://data.laadsdaac.earthdatacloud.nasa.gov`` fails with 401.
Direct S3 access (``s3://prod-lads/...`` with earthaccess temporary
credentials) only works from inside AWS us-west-2.

So we store virtual references with their canonical ``s3://prod-lads/<key>``
URLs (portable: in-region you just point the container at real S3 with
``icechunk.s3_refreshable_credentials``), and out-of-region we point the
container's ``s3_store(endpoint_url=...)`` at this gateway, which serves
path-style ``GET /prod-lads/<key>`` (+ ``Range``) by fetching the same key from
the LAADS HTTPS endpoint with obstore and the EDL token.
"""

from __future__ import annotations

import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import obstore

from .io import STATS, http_store

LAADS = "https://data.laadsdaac.earthdatacloud.nasa.gov"
_RANGE = re.compile(r"bytes=(\d+)-(\d*)")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _store_and_key(self):
        bucket, _, key = self.path.split("?", 1)[0].lstrip("/").partition("/")
        return http_store(f"{self.server.upstream}/{bucket}"), key

    def do_HEAD(self):
        store, key = self._store_and_key()
        meta = obstore.head(store, key)
        self.send_response(200)
        self.send_header("Content-Length", str(meta["size"]))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        store, key = self._store_and_key()
        rng = self.headers.get("Range")
        try:
            if rng:
                m = _RANGE.match(rng)
                start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2)) + 1
                    data = bytes(obstore.get_range(store, key, start=start, end=end))
                else:
                    data = bytes(obstore.get(store, key, options={"range": {"offset": start}}).bytes())
                size = obstore.head(store, key)["size"] if not m.group(2) else None
                self.send_response(206)
                total = "*" if size is None else str(size)
                self.send_header("Content-Range", f"bytes {start}-{start + len(data) - 1}/{total}")
            else:
                data = bytes(obstore.get(store, key).bytes())
                self.send_response(200)
        except FileNotFoundError:
            self.send_error(404)
            return
        STATS.add(1, len(data), 0.0)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()
        self.wfile.write(data)


class Gateway:
    """Run the gateway in a background thread: ``with Gateway() as gw: gw.endpoint``."""

    def __init__(self, host="127.0.0.1", port=0, upstream=LAADS):
        self.server = ThreadingHTTPServer((host, port), _Handler)
        self.server.daemon_threads = True
        self.server.upstream = upstream
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def endpoint(self):
        h, p = self.server.server_address[:2]
        return f"http://{h}:{p}"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
