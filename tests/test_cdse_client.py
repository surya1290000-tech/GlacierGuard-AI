"""
Tests for CDSE STAC Catalogue Client
====================================
Tests for query formatting, feature parsing, retry/failover logic,
and duplicate prevention without making live network calls in CI.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from glacierguard.cdse_client import CDSEClient
from glacierguard.config import Settings
from glacierguard.models import SatelliteProduct


MOCK_STAC_RESPONSE = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "id": "S2A_MSIL2A_20260928T050701_N0511_R019_T45RVJ_20260928T082000",
            "collection": "sentinel-2-l2a",
            "bbox": [86.90, 27.88, 86.96, 27.92],
            "geometry": {"type": "Polygon", "coordinates": [[[86.90, 27.88], [86.96, 27.88], [86.96, 27.92], [86.90, 27.92], [86.90, 27.88]]]},
            "properties": {
                "datetime": "2026-09-28T05:07:01.024Z",
                "platform": "SENTINEL-2A",
                "sat:orbit_state": "DESCENDING",
                "eo:cloud_cover": 0.05,
            },
            "assets": {"visual": {"href": "https://example.com/visual.tif"}},
        },
        {
            "type": "Feature",
            "id": "S1A_IW_GRDH_1SDV_20260928T121000_066490_07F8A1_4123",
            "collection": "sentinel-1-grd",
            "bbox": [86.90, 27.88, 86.96, 27.92],
            "geometry": {"type": "Polygon", "coordinates": [[[86.90, 27.88], [86.96, 27.88], [86.96, 27.92], [86.90, 27.92], [86.90, 27.88]]]},
            "properties": {
                "datetime": "2026-09-28T12:10:00.000Z",
                "platform": "SENTINEL-1A",
                "sat:orbit_state": "ASCENDING",
                "cloudCover": None,
            },
            "assets": {"vv": {"href": "https://example.com/vv.tif"}},
        },
    ],
}


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Fixture providing isolated settings with a temporary database."""
    settings = Settings()
    settings.db_path = tmp_path / "test_client.db"
    settings.stac_timeout_seconds = 2
    settings.stac_max_retries = 2
    return settings


class TestCDSEClientParsing:
    """Test parsing of raw STAC features into domain objects."""

    def test_parse_stac_optical_feature(self, test_settings: Settings):
        client = CDSEClient(test_settings)
        feat = MOCK_STAC_RESPONSE["features"][0]
        prod = client.parse_stac_feature(feat, lake_id="imja_tsho")

        assert isinstance(prod, SatelliteProduct)
        assert prod.product_id == "S2A_MSIL2A_20260928T050701_N0511_R019_T45RVJ_20260928T082000"
        assert prod.collection == "sentinel-2-l2a"
        assert prod.lake_id == "imja_tsho"
        assert prod.platform == "SENTINEL-2A"
        assert prod.orbit_direction == "DESCENDING"
        assert prod.cloud_cover_pct == pytest.approx(0.05)
        assert "visual" in prod.metadata["assets"]

    def test_parse_stac_sar_feature(self, test_settings: Settings):
        client = CDSEClient(test_settings)
        feat = MOCK_STAC_RESPONSE["features"][1]
        prod = client.parse_stac_feature(feat, lake_id="imja_tsho")

        assert prod.product_id.startswith("S1A_IW_GRDH")
        assert prod.collection == "sentinel-1-grd"
        assert prod.platform == "SENTINEL-1A"
        assert prod.orbit_direction == "ASCENDING"
        assert prod.cloud_cover_pct is None


class TestCDSEClientQueries:
    """Test STAC querying, retries, and endpoint failover."""

    @patch("urllib.request.urlopen")
    def test_search_stac_success(self, mock_urlopen, test_settings: Settings):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(MOCK_STAC_RESPONSE).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = CDSEClient(test_settings)
        res = client.search_stac(
            collection="sentinel-2-l2a",
            bbox=[86.90, 27.88, 86.96, 27.92],
            datetime_range="2026-09-01T00:00:00Z/2026-09-30T00:00:00Z",
        )

        assert "features" in res
        assert len(res["features"]) == 2
        assert mock_urlopen.call_count == 1

    @patch("urllib.request.urlopen")
    def test_search_stac_retry_and_failover(self, mock_urlopen, test_settings: Settings):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(MOCK_STAC_RESPONSE).encode("utf-8")
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_resp

        mock_urlopen.side_effect = [
            urllib.error.URLError("Connection refused"),
            urllib.error.URLError("Timeout"),
            mock_cm,
        ]

        client = CDSEClient(test_settings)
        res = client.search_stac(
            collection="sentinel-2-l2a",
            bbox=[86.90, 27.88, 86.96, 27.92],
            datetime_range="2026-09-01T00:00:00Z/2026-09-30T00:00:00Z",
        )

        assert "features" in res
        assert mock_urlopen.call_count >= 3

    @patch("urllib.request.urlopen")
    def test_search_stac_complete_failure_raises(self, mock_urlopen, test_settings: Settings):
        mock_urlopen.side_effect = urllib.error.URLError("Host unreachable")

        client = CDSEClient(test_settings)
        with pytest.raises(RuntimeError, match="Failed to query STAC catalogue"):
            client.search_stac(
                collection="sentinel-2-l2a",
                bbox=[86.90, 27.88, 86.96, 27.92],
                datetime_range="2026-09-01T00:00:00Z/2026-09-30T00:00:00Z",
            )


class TestCDSEClientPersistence:
    """Test discovery and database deduplication."""

    @patch.object(CDSEClient, "search_stac")
    def test_discover_and_persist_deduplicates(self, mock_search, test_settings: Settings):
        mock_search.return_value = MOCK_STAC_RESPONSE
        client = CDSEClient(test_settings)

        # First run: inserts products
        new_count_1, products_1 = client.discover_and_persist(
            lake_id="imja_tsho",
            start_date="2026-09-01",
            end_date="2026-09-30",
        )
        assert new_count_1 == 2
        assert len(products_1) == 4  # 2 collections x 2 features

        # Second run: duplicate products should be ignored
        new_count_2, products_2 = client.discover_and_persist(
            lake_id="imja_tsho",
            start_date="2026-09-01",
            end_date="2026-09-30",
        )
        assert new_count_2 == 0  # Deduplicated!
