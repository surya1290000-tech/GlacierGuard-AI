"""
Unit tests for Phase 9.1.2: Nepal 2026 Cryospheric Event Schema & Spatial Audit
=============================================================================
Tests verify event schema compliance, hazard type taxonomy, spatial audit
integrity, satellite availability reporting, and scientific guardrails.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENTS_JSON = REPO_ROOT / "data" / "events" / "nepal_2026" / "event_catalogue.json"
SPATIAL_AUDIT_JSON = REPO_ROOT / "reports" / "13_event_spatial_coverage_audit.json"
SCENES_CSV = REPO_ROOT / "reports" / "13_satellite_scene_availability.csv"
FROZEN_MODEL_PKL = REPO_ROOT / "reports" / "models" / "09_final_fused_xgboost.pkl"


class TestEventCatalogueSchema:
    """Verify event catalogue JSON structure, taxonomy, and scientific guardrails."""

    def test_catalogue_file_exists_and_valid_json(self):
        assert EVENTS_JSON.is_file(), f"Missing {EVENTS_JSON}"
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("schema_version") in ["1.0.0", "1.1.0"]
        assert len(data.get("events", [])) == 2

    def test_canonical_hazard_types_enforced(self):
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        allowed_types = set(data.get("allowed_primary_hazard_types", []))
        expected_types = {
            "ROCK_ICE_AVALANCHE",
            "GLACIAL_LAKE_OUTBURST",
            "LANDSLIDE_DAM_BREACH",
            "DEBRIS_FLOW",
            "DOWNSTREAM_FLOOD",
            "TEMPORARY_RIVER_BLOCKAGE",
        }
        assert expected_types == allowed_types

        for event in data["events"]:
            assert event["primary_type"] in allowed_types
            for casc in event.get("cascading_types", []):
                assert casc in allowed_types

    def test_langtang_event_specification(self):
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        events = {e["event_id"]: e for e in data["events"]}
        assert "NEPAL-2026-08-26-LANGTANG" in events
        langtang = events["NEPAL-2026-08-26-LANGTANG"]

        assert langtang["event_date"] == "2026-08-26"
        assert langtang["primary_type"] == "ROCK_ICE_AVALANCHE"
        assert langtang["is_provisional"] is False
        assert langtang["mechanism_confidence"] == "HIGH"
        assert "Lhende Khola" in langtang["catchment"]
        assert langtang["source_location"]["latitude"] == pytest.approx(28.288708, abs=1e-5)
        assert langtang["source_location"]["longitude"] == pytest.approx(85.528159, abs=1e-5)

        # Scientific guardrails
        ml_rel = langtang["relationship_to_existing_ml"]
        assert ml_rel["is_glo_lake_expansion"] is False
        assert ml_rel["is_in_glo_lake_training_footprint"] is False
        assert ml_rel["predictable_by_fused_xgboost_regressor"] is False

    def test_budhi_gandaki_event_specification(self):
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        events = {e["event_id"]: e for e in data["events"]}
        assert "NEPAL-2026-10-11-BUDHI-GANDAKI" in events
        budhi = events["NEPAL-2026-10-11-BUDHI-GANDAKI"]

        assert budhi["event_date"] == "2026-10-11"
        assert budhi["primary_type"] in ["LANDSLIDE_DAM_BREACH", "ROCK_ICE_AVALANCHE"]
        if budhi["primary_type"] == "LANDSLIDE_DAM_BREACH":
            assert budhi["is_provisional"] is True
            assert "PROVISIONAL" in budhi["mechanism_confidence"]
        else:
            assert budhi["primary_type"] == "ROCK_ICE_AVALANCHE"
            assert any(
                r.get("revised_primary_type") == "LANDSLIDE_DAM_BREACH"
                or r.get("prior_primary_type") == "LANDSLIDE_DAM_BREACH"
                for r in budhi.get("revision_history", [])
            )
        assert "Budhi Gandaki" in budhi["catchment"]

        # Scientific guardrails
        ml_rel = budhi["relationship_to_existing_ml"]
        assert ml_rel["is_glo_lake_expansion"] is False
        assert ml_rel["is_in_glo_lake_training_footprint"] is False


class TestSpatialCoverageAudit:
    """Verify spatial calculations, buffer searches, and lack of training overlap."""

    def test_spatial_audit_file_valid(self):
        assert SPATIAL_AUDIT_JSON.is_file()
        with open(SPATIAL_AUDIT_JSON, "r", encoding="utf-8") as f:
            audit = json.load(f)
        assert "langtang_lirung_event" in audit
        assert "budhi_gandaki_event" in audit

    def test_source_locations_not_inside_glo_lake_polygons(self):
        with open(SPATIAL_AUDIT_JSON, "r", encoding="utf-8") as f:
            audit = json.load(f)
        assert audit["langtang_lirung_event"]["point_in_glo_lake_polygon"] is False
        assert audit["budhi_gandaki_event"]["point_in_glo_lake_polygon"] is False

    def test_langtang_nearest_lake_and_buffer_counts(self):
        with open(SPATIAL_AUDIT_JSON, "r", encoding="utf-8") as f:
            audit = json.load(f)
        lt = audit["langtang_lirung_event"]
        # Nearest GLO lake is > 3.0 km away
        assert lt["nearest_glo_s2_lake"]["distance_m"] > 3000.0
        assert lt["nearest_glo_s2_lake"]["distance_km"] == pytest.approx(3.68, abs=0.1)
        # Buffer counts: 5km = 1, 10km = 14, 20km = 117
        radii = lt["glo_lake_counts_within_radii"]
        assert radii["within_5km"] == 1
        assert radii["within_10km"] == 14
        assert radii["within_20km"] >= 100

    def test_training_feature_exclusion_acknowledged(self):
        with open(SPATIAL_AUDIT_JSON, "r", encoding="utf-8") as f:
            audit = json.load(f)
        rep = audit["representation_in_ml_training_features"]
        assert rep["is_source_location_in_training_data"] is False
        assert rep["is_affected_corridor_in_training_data"] is False


class TestSatelliteSceneAvailability:
    """Verify scene catalogue parsing, provenance, and status distinction."""

    def test_scenes_csv_exists_and_has_records(self):
        assert SCENES_CSV.is_file()
        df = pd.read_csv(SCENES_CSV)
        assert len(df) > 0

    def test_scenes_cover_pre_and_post_event(self):
        df = pd.read_csv(SCENES_CSV)
        periods = set(df["period"].unique())
        assert "PRE_EVENT" in periods
        assert "POST_EVENT" in periods

    def test_scenes_contain_both_sar_and_optical(self):
        df = pd.read_csv(SCENES_CSV)
        collections = set(df["collection"].unique())
        assert "sentinel-1-slc" in collections
        assert "sentinel-1-grd" in collections
        assert "sentinel-2-l2a" in collections

    def test_status_distinguishes_discovery_from_processing(self):
        df = pd.read_csv(SCENES_CSV)
        for status in df["processing_status"].unique():
            assert "DISCOVERED_IN_CATALOGUE" in status
            assert "PROCESSED" not in status  # Raw scenes are NOT claimed to be processed


class TestScientificGuardrailsAndSecurity:
    """Ensure frozen model, secrets, and safety guardrails remain intact."""

    def test_frozen_model_is_unmodified(self):
        assert FROZEN_MODEL_PKL.is_file()
        import pickle
        with open(FROZEN_MODEL_PKL, "rb") as f:
            model_dict = pickle.load(f)
        assert model_dict["model_class"] == "XGBRegressor"
        assert model_dict["test_mae"] == pytest.approx(0.004792, abs=1e-5)
        assert model_dict["feature_count_raw"] == 36

    def test_no_raw_credentials_in_event_catalogue(self):
        content = EVENTS_JSON.read_text(encoding="utf-8")
        assert "client_secret" not in content.lower()
        assert "password" not in content.lower()
        assert "bearer" not in content.lower()
