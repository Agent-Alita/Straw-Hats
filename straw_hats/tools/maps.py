"""Nominatim (OpenStreetMap) geocoding and POI search, biased to San Francisco."""
from __future__ import annotations

from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential

from ._common import RateLimiter, cached_tool, err, http_session, ok


# Nominatim usage policy: max 1 req/sec
_LIMITER = RateLimiter(min_interval_s=1.1)

# SF bounding box (lon_min, lat_min, lon_max, lat_max) — Nominatim viewbox is lon,lat,lon,lat
SF_VIEWBOX = "-122.5500,37.8330,-122.3500,37.7000"
SF_BBOX = (37.70, -122.55, 37.83, -122.35)  # (lat_min, lon_min, lat_max, lon_max)


def _in_sf(lat: float, lng: float) -> bool:
    return SF_BBOX[0] <= lat <= SF_BBOX[2] and SF_BBOX[1] <= lng <= SF_BBOX[3]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def _nominatim(path: str, params: dict) -> list | dict:
    _LIMITER.wait()
    s = http_session()
    r = s.get(f"https://nominatim.openstreetmap.org/{path}", params=params, timeout=25)
    if r.status_code == 429:
        raise RuntimeError("nominatim rate limited")
    r.raise_for_status()
    return r.json()


def _format_result(item: dict) -> dict:
    try:
        lat = float(item.get("lat"))
        lng = float(item.get("lon"))
    except (TypeError, ValueError):
        lat = lng = None
    return {
        "display_name": item.get("display_name"),
        "lat": lat,
        "lng": lng,
        "type": item.get("type"),
        "class": item.get("class"),
        "importance": item.get("importance"),
        "osm_id": item.get("osm_id"),
        "in_sf": _in_sf(lat, lng) if lat is not None else False,
    }


@tool
@cached_tool(ttl_seconds=30 * 24 * 3600)  # 30 days
def geocode(query: str, limit: int = 5) -> dict:
    """Forward-geocode an address, place name, or landmark, biased to San Francisco.
    Returns up to `limit` candidates with lat/lng and a display name.

    Args:
        query: free-form text, e.g. "Buena Vista Park" or "1 Sansome St, San Francisco".
        limit: max results (1-10).
    """
    if not query.strip():
        return err("empty query")
    try:
        results = _nominatim(
            "search",
            {
                "q": query,
                "format": "jsonv2",
                "limit": max(1, min(10, int(limit))),
                "viewbox": SF_VIEWBOX,
                "bounded": 0,
                "addressdetails": 1,
                "countrycodes": "us",
            },
        )
    except Exception as e:  # noqa: BLE001
        return err(f"geocode failed: {e}")

    items = [_format_result(it) for it in (results or [])]
    # Bias: sort SF-internal results to the top
    items.sort(key=lambda x: (not x["in_sf"], -(x["importance"] or 0)))
    return ok({"query": query, "results": items})


@tool
@cached_tool(ttl_seconds=30 * 24 * 3600)  # 30 days
def reverse_geocode(lat: float, lng: float) -> dict:
    """Reverse-geocode a coordinate to a street address / place description.

    Args:
        lat: latitude.
        lng: longitude.
    """
    try:
        result = _nominatim(
            "reverse",
            {"lat": lat, "lon": lng, "format": "jsonv2", "addressdetails": 1, "zoom": 18},
        )
    except Exception as e:  # noqa: BLE001
        return err(f"reverse geocode failed: {e}")
    return ok(
        {
            "display_name": result.get("display_name"),
            "address": result.get("address"),
            "lat": float(result.get("lat", lat)),
            "lng": float(result.get("lon", lng)),
        }
    )


def _bbox_around(lat: float, lng: float, radius_m: float) -> str:
    # rough: 1 deg lat = 111_111 m
    dlat = radius_m / 111_111.0
    # 1 deg lon at SF latitude ~ 87_834 m
    dlon = radius_m / 87_834.0
    return f"{lng - dlon:.6f},{lat + dlat:.6f},{lng + dlon:.6f},{lat - dlat:.6f}"


@tool
@cached_tool(ttl_seconds=30 * 24 * 3600)  # 30 days
def nearby_search(lat: float, lng: float, query: str, radius_m: int = 400) -> dict:
    """Search for points of interest near a coordinate (within radius_m meters).
    Use to refine a neighborhood guess to a specific bench, statue, plaque, tree,
    fountain, monument, mural, sign, etc.

    Args:
        lat: latitude of the center point.
        lng: longitude of the center point.
        query: feature to search for, e.g. "plaque", "statue", "fountain", "bench".
        radius_m: search radius in meters (50-2000).
    """
    radius_m = max(50, min(2000, int(radius_m)))
    viewbox = _bbox_around(lat, lng, radius_m)
    try:
        results = _nominatim(
            "search",
            {
                "q": query,
                "format": "jsonv2",
                "limit": 10,
                "viewbox": viewbox,
                "bounded": 1,
                "addressdetails": 1,
            },
        )
    except Exception as e:  # noqa: BLE001
        return err(f"nearby search failed: {e}")
    items = [_format_result(it) for it in (results or [])]
    return ok({"center": {"lat": lat, "lng": lng}, "radius_m": radius_m, "query": query, "results": items})
