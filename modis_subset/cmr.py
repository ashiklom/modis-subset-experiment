"""CMR granule search (earthaccess) and product/geolocation pairing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .io import login

_KEY = re.compile(r"\.A(\d{7})\.(\d{4})\.")


@dataclass
class Granule:
    short_name: str
    url: str
    key: str                 # "YYYYDDD.HHMM" -- identical for a product and its MOD03
    begin: datetime
    end: datetime
    orbit: int | None
    eq_lon: float | None     # equator crossing longitude (deg)
    eq_time: datetime | None # equator crossing time
    day_night: str | None
    size_mb: float | None
    polygon: list | None = None  # [(lon, lat), ...] CMR GPolygon (granule corners)

    @property
    def name(self):
        return self.url.rsplit("/", 1)[-1]


def _dt(s):
    if s is None:
        return None
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _parse(g) -> Granule | None:
    umm = g["umm"]
    url = next(
        (u for u in g.data_links(access="external") if u.endswith((".hdf", ".nc"))),
        None,
    )
    if url is None:
        return None
    m = _KEY.search(url)
    rng = umm["TemporalExtent"]["RangeDateTime"]
    orb = (umm.get("OrbitCalculatedSpatialDomains") or [{}])[0]
    return Granule(
        short_name=umm["CollectionReference"]["ShortName"],
        url=url,
        key=f"{m.group(1)}.{m.group(2)}",
        begin=_dt(rng["BeginningDateTime"]),
        end=_dt(rng["EndingDateTime"]),
        orbit=orb.get("OrbitNumber"),
        eq_lon=orb.get("EquatorCrossingLongitude"),
        eq_time=_dt(orb.get("EquatorCrossingDateTime")),
        day_night=umm.get("DataGranule", {}).get("DayNightFlag"),
        size_mb=g["size"] if "size" in g else None,
        polygon=_polygon(umm),
    )


def _polygon(umm):
    try:
        geom = umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
        pts = geom["GPolygons"][0]["Boundary"]["Points"]
        return [(p["Longitude"], p["Latitude"]) for p in pts]
    except (KeyError, IndexError):
        return None


def search(short_name, lat, lon, start, end, version="6.1"):
    """Granules of ``short_name`` whose footprint contains (lat, lon)."""
    import earthaccess

    login()
    res = earthaccess.search_data(
        short_name=short_name, version=version, point=(lon, lat), temporal=(start, end),
    )
    out = [g for g in map(_parse, res) if g is not None]
    return sorted(out, key=lambda g: g.begin)


def geolocation_short_name(short_name: str) -> str:
    """MOD35_L2 -> MOD03, MYD06_L2 -> MYD03."""
    return short_name[:3] + "03"


def paired(short_name, lat, lon, start, end, version="6.1"):
    """[(product_granule, geolocation_granule)] matched on the A-date/time key."""
    prod = search(short_name, lat, lon, start, end, version)
    geo = {g.key: g for g in search(geolocation_short_name(short_name), lat, lon, start, end, version)}
    return [(p, geo.get(p.key)) for p in prod]
