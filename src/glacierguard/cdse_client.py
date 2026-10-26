"""
CDSE STAC Catalogue Client
==========================
Client for querying Copernicus Data Space Ecosystem (CDSE) STAC catalogue
for Sentinel-1 GRD and Sentinel-2 L2A observations across monitored lakes.

Features:
- Supports both primary (`/v1`) and catalogue fallback STAC endpoints.
- Unauthenticated catalogue search (no credentials required for metadata).
- Automatic retries with exponential backoff and timeout safeguards.
- Deduplication against SQLite database before persistence.
- Zero external HTTP dependencies (uses Python standard library ``urllib``).
"""

import datetime
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

from glacierguard.config import Settings, default_settings
from glacierguard.db import product_exists, save_satellite_products
from glacierguard.models import SatelliteProduct


logger = logging.getLogger(__name__)


class CDSEClient:
    """Client for querying the Copernicus Data Space Ecosystem STAC API."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or default_settings
        self.primary_url = self.settings.stac_api_url.rstrip("/")
        self.fallback_url = self.settings.fallback_stac_api_url.rstrip("/")
        self.timeout = self.settings.stac_timeout_seconds
        self.max_retries = self.settings.stac_max_retries

    def _execute_query(
        self,
        base_url: str,
        collection: str,
        bbox: List[float],
        datetime_range: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Perform a single HTTP POST request to a STAC /search endpoint."""
        url = f"{base_url}/search"
        payload = {
            "collections": [collection],
            "bbox": bbox,
            "datetime": datetime_range,
            "limit": limit,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "User-Agent": "GlacierGuard-AI/0.1.0",
                "Content-Type": "application/json",
                "Accept": "application/geo+json, application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content)

    def search_stac(
        self,
        collection: str,
        bbox: List[float],
        datetime_range: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Search STAC catalogue with retries and fallback endpoint failover.

        Parameters
        ----------
        collection : str
            STAC collection identifier (e.g., 'sentinel-1-grd', 'sentinel-2-l2a').
        bbox : List[float]
            Bounding box [min_lon, min_lat, max_lon, max_lat].
        datetime_range : str
            ISO 8601 interval string (e.g., '2026-09-01T00:00:00Z/2026-09-30T23:59:59Z').
        limit : int, default=20
            Maximum number of features to retrieve.

        Returns
        -------
        Dict[str, Any]
            GeoJSON FeatureCollection dictionary.

        Raises
        ------
        RuntimeError
            If both primary and fallback endpoints fail after retries.
        """
        endpoints = [self.primary_url, self.fallback_url]
        last_error: Optional[Exception] = None

        for endpoint in endpoints:
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.debug(
                        "Querying STAC %s (attempt %d/%d) for %s",
                        endpoint,
                        attempt,
                        self.max_retries,
                        collection,
                    )
                    return self._execute_query(
                        base_url=endpoint,
                        collection=collection,
                        bbox=bbox,
                        datetime_range=datetime_range,
                        limit=limit,
                    )
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    last_error = exc
                    logger.warning(
                        "STAC query to %s failed (attempt %d/%d): %s",
                        endpoint,
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    if attempt < self.max_retries:
                        time.sleep(0.5 * (2 ** (attempt - 1)))  # Exponential backoff
                except Exception as exc:
                    last_error = exc
                    logger.error("Unexpected error querying STAC %s: %s", endpoint, exc)
                    break  # Fail over to next endpoint

        raise RuntimeError(
            f"Failed to query STAC catalogue across endpoints {endpoints}: {last_error}"
        )

    def parse_stac_feature(
        self,
        feature: Dict[str, Any],
        lake_id: Optional[str] = None,
    ) -> SatelliteProduct:
        """Parse a STAC GeoJSON feature into a SatelliteProduct domain object."""
        product_id = feature.get("id", "UNKNOWN_PRODUCT")
        collection = feature.get("collection", "UNKNOWN_COLLECTION")
        properties = feature.get("properties", {})
        bbox = feature.get("bbox", [])

        # Datetime parsing
        dt_val = (
            properties.get("datetime")
            or properties.get("start_datetime")
            or properties.get("end_datetime")
            or datetime.datetime.utcnow().isoformat() + "Z"
        )

        # Cloud cover
        cloud = properties.get("eo:cloud_cover")
        if cloud is None:
            cloud = properties.get("cloudCover")

        # Platform and orbit
        platform = properties.get("platform") or properties.get("constellation")
        orbit = properties.get("sat:orbit_state") or properties.get("orbitDirection")

        return SatelliteProduct(
            product_id=product_id,
            collection=collection,
            acquisition_date=dt_val,
            bbox=bbox,
            lake_id=lake_id,
            cloud_cover_pct=float(cloud) if cloud is not None else None,
            platform=str(platform) if platform else None,
            orbit_direction=str(orbit) if orbit else None,
            discovered_at=datetime.datetime.utcnow().isoformat() + "Z",
            metadata={
                "geometry": feature.get("geometry"),
                "assets": list(feature.get("assets", {}).keys()),
            },
        )

    def discover_products_for_lake(
        self,
        lake_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        collections: Optional[List[str]] = None,
        limit_per_collection: int = 10,
    ) -> List[SatelliteProduct]:
        """Discover satellite products for a monitored lake over a time window.

        Parameters
        ----------
        lake_id : str
            Lake identifier (e.g., 'imja_tsho').
        start_date : str, optional
            Start date 'YYYY-MM-DD' (defaults to 30 days ago).
        end_date : str, optional
            End date 'YYYY-MM-DD' (defaults to today).
        collections : List[str], optional
            Collections to query (defaults to configured collections).
        limit_per_collection : int, default=10
            Max scenes per collection.

        Returns
        -------
        List[SatelliteProduct]
            List of parsed satellite products sorted by acquisition date descending.
        """
        lake_cfg = self.settings.get_lake_config(lake_id)
        target_collections = collections or self.settings.collections

        # Date formatting
        today = datetime.date.today()
        if not end_date:
            end_date = today.isoformat()
        if not start_date:
            start_date = (today - datetime.timedelta(days=30)).isoformat()

        dt_range = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"

        all_products: List[SatelliteProduct] = []

        for coll in target_collections:
            try:
                resp = self.search_stac(
                    collection=coll,
                    bbox=lake_cfg.bbox,
                    datetime_range=dt_range,
                    limit=limit_per_collection,
                )
                features = resp.get("features", [])
                for feat in features:
                    prod = self.parse_stac_feature(feat, lake_id=lake_id)
                    all_products.append(prod)
            except Exception as exc:
                logger.warning(
                    "Error searching %s for lake %s: %s", coll, lake_id, exc
                )

        # Sort descending by date
        all_products.sort(key=lambda p: p.acquisition_date, reverse=True)
        return all_products

    def discover_and_persist(
        self,
        lake_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[int, List[SatelliteProduct]]:
        """Discover products and save newly discovered products to database.

        Returns
        -------
        Tuple[int, List[SatelliteProduct]]
            (number of newly added products, all discovered products)
        """
        products = self.discover_products_for_lake(
            lake_id=lake_id,
            start_date=start_date,
            end_date=end_date,
        )

        new_products = [
            p for p in products if not product_exists(p.product_id, self.settings.db_path)
        ]
        inserted = save_satellite_products(new_products, self.settings.db_path)
        return inserted, products
