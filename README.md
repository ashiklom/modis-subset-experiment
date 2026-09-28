# MODIS level-2 point time series

Extract a time series at one lat/lon point from MODIS level-2 swath products
(e.g. `MOD35_L2` cloud mask, `MOD04_L2` aerosol), and compare several ways of
making the geolocation and data-reading steps cheap.

* NASA credentials and CMR queries: `earthaccess` (uses `EARTHDATA_LOGIN_USER`
  / `EARTHDATA_LOGIN_PASSWORD`, mapped onto earthaccess' variable names).
* All byte I/O: `obstore` (local files and HTTPS with the EDL bearer token).
* Virtual references: our own HDF4 scanner (on top of kerchunk's), persisted
  as **Parquet**, read with obstore range requests.

See [`REPORT.md`](REPORT.md) for the experiment results and what did / did not
work.

## Quick start

```bash
uv sync
export EARTHDATA_LOGIN_USER=... EARTHDATA_LOGIN_PASSWORD=...
uv run python - <<'EOF'
from modis_subset.timeseries import extract
df = extract("MOD35_L2", 41.31, -72.92, "2024-01-01", "2024-01-31T23:59:59",
             ["Cloud_Mask"], strategy="interp5km")
print(df[["time", "day_night", "row", "col", "dist_km", "cm_cloudiness"]])
EOF
```

`strategy` is one of `native`, `interp5km`, `brute`, `analytic_window`,
`index_window` (see `modis_subset/timeseries.py`). Pass `local_dir=` to read
already-downloaded files (`<local_dir>/<short_name>/<file>`) instead of HTTPS.

## Layout

| module | what |
|---|---|
| `modis_subset/io.py` | earthaccess login, obstore `Source` (local / HTTPS), block-cached file object, I/O counters |
| `modis_subset/cmr.py` | granule search, product <-> MOD03 pairing, CMR footprint polygon |
| `modis_subset/hdf4.py` | HDF4 SDS byte-range references (fixes kerchunk's HDF4 float byte-order bug; handles chunked MOD03 arrays) |
| `modis_subset/refcache.py` | Parquet reference cache (one row per chunk) |
| `modis_subset/reader.py` | chunk-aware reads; `read_points` = partial zlib decompression of single-block variables |
| `modis_subset/geo.py` | ECEF nearest-pixel search |
| `modis_subset/orbit.py` | analytic (row, col) prediction from the CMR footprint |
| `modis_subset/locate.py` | locator strategies (brute, windowed MOD03 search, 5 km -> 1 km interpolation) |
| `modis_subset/index.py` | cached spatial index (S2-style cube-face cells -> scan range), Parquet |
| `modis_subset/timeseries.py` | `extract()` end-to-end pipeline |
| `scripts/` | download, validation and benchmark scripts used for `REPORT.md` |
