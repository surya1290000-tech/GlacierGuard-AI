"""
Unit tests for Phase 9.1.3: Event Evidence Reconciliation & Spatial Audit Hardening
===================================================================================
Tests verify:
1. Budhi Gandaki primary classification as ROCK_ICE_AVALANCHE and cascade sequence.
2. Preservation of 11 Oct ICIMOD preliminary report in revision history.
3. No false assertion of Chumchet impact coordinates as exact avalanche source.
4. Unresolved source coordinates represented as null.
5. TEMPORARY_RIVER_BLOCKAGE recognized as distinct allowed event type.
6. Langtang source point classified as SPATIALLY_AMBIGUOUS_WITHIN_REPORTED_PRECISION.
7. RGI polygon metadata/provenance required (e.g. RGI 6.0, not 2026 ground truth).
8. S1 acquisition grouping retains both SLC and GRD products within each group.
9. S2 multi-tile acquisition grouping correctly preserves multiple granules.
10. Product totals in updated catalogue reconcile with product-level CSVs.
11. Acquisition group totals in summary reconcile with independent acquisition counts.
12. Langtang and Budhi Gandaki satellite inventories have separate AOI identifiers.
13. Distinction between scene discovery and raster processing maintained across records.
14. Event records cannot be integrated into GLO lake regression training targets.
15. Frozen XGBoost model SHA-256 matches frozen baseline byte-for-byte.
16. No raw credentials or tokens appear in any report, CSV, or JSON output.
"""

import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENTS_JSON = REPO_ROOT / "data" / "events" / "nepal_2026" / "event_catalogue.json"
SPATIAL_AUDIT_JSON = REPO_ROOT / "reports" / "13_event_spatial_coverage_audit.json"
LANGTANG_SCENES_CSV = REPO_ROOT / "reports" / "13_satellite_scene_availability.csv"
LANGTANG_ACQ_CSV = REPO_ROOT / "reports" / "13_acquisition_group_summary.csv"
BUDHI_SCENES_CSV = REPO_ROOT / "reports" / "13_budhi_gandaki_scene_availability.csv"
RECON_MD = REPO_ROOT / "reports" / "13_event_evidence_reconciliation.md"
MULTI_HAZARD_MD = REPO_ROOT / "reports" / "13_multi_hazard_model_design.md"
FROZEN_MODEL_PKL = REPO_ROOT / "reports" / "models" / "09_final_fused_xgboost.pkl"

EXPECTED_FROZEN_SHA256 = "5c5bec942c316a8c4a3f40d7098f7d50a4d7ac34d0f4ba5e308c7d19fc75525f"


class TestBudhiGandakiReconciliation:
    """Tests 1-4: Budhi Gandaki hazard reclassification, evidence provenance, and coordinates."""

    def test_01_budhi_gandaki_classification_and_cascade(self):
        """1. Budhi Gandaki primary classified as ROCK_ICE_AVALANCHE with exact cascade sequence."""
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        events = {e["event_id"]: e for e in data["events"]}
        budhi = events["NEPAL-2026-10-11-BUDHI-GANDAKI"]

        assert budhi["primary_type"] == "ROCK_ICE_AVALANCHE"
        expected_cascade = ["DEBRIS_FLOW", "TEMPORARY_RIVER_BLOCKAGE", "DOWNSTREAM_FLOOD"]
        assert budhi["cascading_types"] == expected_cascade

        # Verify sequential cascade stage details
        stages = [s["hazard_process"] for s in budhi.get("cascade_stage_details", [])]
        assert stages == ["ROCK_ICE_AVALANCHE", "DEBRIS_FLOW", "TEMPORARY_RIVER_BLOCKAGE", "DOWNSTREAM_FLOOD"]

    def test_02_preliminary_icimod_preserved_in_revision_history(self):
        """2. 11 October ICIMOD preliminary report preserved in revision history."""
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        budhi = next(e for e in data["events"] if e["event_id"] == "NEPAL-2026-10-11-BUDHI-GANDAKI")

        rev_history = budhi.get("revision_history", [])
        assert len(rev_history) >= 2

        # Check initial revision preserves 11 Oct report
        init_rev = rev_history[0]
        assert init_rev["revision_date"] == "2026-10-11"
        assert init_rev["revised_primary_type"] == "LANDSLIDE_DAM_BREACH"
        assert "landslide" in init_rev["reason"].lower()
        assert "icimod" in init_rev["source_ref"].lower()

        # Check evidence sources contain preliminary superseded tag
        evidence = budhi.get("evidence_sources_comprehensive", [])
        oct11 = next(
            s for s in evidence
            if s.get("publication_date") == "2026-10-11" and s.get("source_org") == "ICIMOD"
        )
        assert oct11["confidence"] == "PRELIMINARY_SUPERSEDED"

    def test_03_no_false_source_coordinate_assertion_for_chumchet(self):
        """3. No assertion made that Chumchet impact coordinates are exact source coordinates."""
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        budhi = next(e for e in data["events"] if e["event_id"] == "NEPAL-2026-10-11-BUDHI-GANDAKI")

        # Top-level source_location coordinates must NOT be Chumchet
        assert budhi["source_location"]["latitude"] is None
        assert budhi["source_location"]["longitude"] is None

        # Chumchet coordinates must be explicitly marked as affected reach
        aff = budhi.get("affected_reach_location", {})
        assert aff.get("latitude") == pytest.approx(28.457, abs=1e-3)
        assert aff.get("longitude") == pytest.approx(84.975, abs=1e-3)
        assert "NOT the avalanche source" in aff.get("description", "")

        # Spatial audit must also document Chumchet as affected reach only
        with open(SPATIAL_AUDIT_JSON, "r", encoding="utf-8") as f:
            sp_data = json.load(f)
        bg_sp = sp_data["budhi_gandaki_event"]["location_clarification"]
        assert bg_sp["chumchet_role"] == "AFFECTED_REACH_REFERENCE_ONLY"

    def test_04_unresolved_source_coordinates_are_null(self):
        """4. Unresolved source coordinates represented as null."""
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        budhi = next(e for e in data["events"] if e["event_id"] == "NEPAL-2026-10-11-BUDHI-GANDAKI")

        assert budhi["source_location"]["latitude"] is None
        assert budhi["source_location"]["longitude"] is None
        assert budhi["location_components"]["source_region"]["coordinates"] is None
        assert "UNRESOLVED" in budhi["location_components"]["source_region"]["spatial_precision"]

        with open(SPATIAL_AUDIT_JSON, "r", encoding="utf-8") as f:
            sp_data = json.load(f)
        assert sp_data["budhi_gandaki_event"]["location_clarification"]["source_coordinates"] is None


class TestTaxonomyAndSpatialAmbiguity:
    """Tests 5-7: Taxonomy extensions, spatial ambiguity handling, and RGI metadata."""

    def test_05_temporary_river_blockage_allowed_type(self):
        """5. TEMPORARY_RIVER_BLOCKAGE recognized as allowed event type distinct from landslide dam."""
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        allowed = set(data["allowed_primary_hazard_types"])
        assert "TEMPORARY_RIVER_BLOCKAGE" in allowed
        assert "LANDSLIDE_DAM_BREACH" in allowed
        assert "TEMPORARY_RIVER_BLOCKAGE" != "LANDSLIDE_DAM_BREACH"

    def test_06_langtang_spatially_ambiguous_within_precision(self):
        """6. Langtang source point classified as SPATIALLY_AMBIGUOUS_WITHIN_REPORTED_PRECISION."""
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        langtang = next(e for e in data["events"] if e["event_id"] == "NEPAL-2026-08-26-LANGTANG")
        recon = langtang["spatial_uncertainty_reconciliation"]

        assert recon["spatial_status"] == "SPATIALLY_AMBIGUOUS_WITHIN_REPORTED_PRECISION"
        dist = recon["distance_to_nearest_rgi_glacier_m"]
        prec = recon["reported_coordinate_precision_m"]
        assert dist == pytest.approx(42.4, abs=0.1)
        assert prec == pytest.approx(150.0, abs=0.1)
        assert dist < prec, "Distance to glacier must be strictly smaller than coordinate precision"

        with open(SPATIAL_AUDIT_JSON, "r", encoding="utf-8") as f:
            sp_data = json.load(f)
        lt_sp = sp_data["langtang_lirung_event"]["spatial_ambiguity_assessment"]
        assert lt_sp["status"] == "SPATIALLY_AMBIGUOUS_WITHIN_REPORTED_PRECISION"
        assert lt_sp["nearest_glacier_dist_m"] < lt_sp["reported_precision_m"]

    def test_07_rgi_polygon_metadata_and_provenance(self):
        """7. RGI polygon metadata and provenance required (RGI 6.0, not 2026 ground truth)."""
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        langtang = next(e for e in data["events"] if e["event_id"] == "NEPAL-2026-08-26-LANGTANG")
        rgi_meta = langtang["spatial_uncertainty_reconciliation"]["rgi_metadata"]

        assert "RGI 6.0" in rgi_meta["inventory_version"]
        assert rgi_meta["is_exact_2026_boundary"] is False
        assert "historical" in rgi_meta["notes"].lower()

        with open(SPATIAL_AUDIT_JSON, "r", encoding="utf-8") as f:
            sp_data = json.load(f)
        prov = sp_data["langtang_lirung_event"]["spatial_ambiguity_assessment"]["rgi_provenance"]
        assert prov["version"] == "RGI 6.0"
        assert prov["is_exact_2026_boundary"] is False


class TestSatelliteGroupingAndInventories:
    """Tests 8-13: Acquisition grouping, product counts, AOIs, and discovery vs processing."""

    def test_08_s1_acquisition_grouping_retains_slc_and_grd(self):
        """8. S1 acquisition grouping retains both SLC and GRD products within each group."""
        df = pd.read_csv(LANGTANG_SCENES_CSV)
        s1 = df[df["collection"].str.contains("sentinel-1")]
        assert len(s1) == 46, f"Expected 46 S1 products, found {len(s1)}"

        # Every S1 group must have exactly 2 products: 1 SLC and 1 GRD
        for gid, grp in s1.groupby("acquisition_group_id"):
            colls = set(grp["collection"].tolist())
            assert "sentinel-1-slc" in colls, f"Group {gid} missing SLC"
            assert "sentinel-1-grd" in colls, f"Group {gid} missing GRD"
            assert len(grp) == 2, f"Group {gid} should have exactly 2 products"

    def test_09_s2_multi_tile_acquisition_preserves_multiple_granules(self):
        """9. S2 multi-tile acquisition grouping correctly preserves multiple granules."""
        df = pd.read_csv(LANGTANG_SCENES_CSV)
        s2 = df[df["collection"].str.contains("sentinel-2")]
        assert len(s2) == 46, f"Expected 46 S2 products, found {len(s2)}"

        # Check multi-tile group exists
        grp_counts = s2.groupby("acquisition_group_id").size()
        multi_tile = grp_counts[grp_counts > 1]
        assert len(multi_tile) >= 1
        assert multi_tile.iloc[0] == 2

    def test_10_product_totals_reconcile_with_csv(self):
        """10. Product totals in updated catalogue reconcile with product-level CSVs."""
        df_langtang = pd.read_csv(LANGTANG_SCENES_CSV)
        assert len(df_langtang) == 92, f"Expected 92 Langtang products, found {len(df_langtang)}"

        df_budhi = pd.read_csv(BUDHI_SCENES_CSV)
        assert len(df_budhi) == 50, f"Expected 50 Budhi Gandaki products, found {len(df_budhi)}"

    def test_11_acquisition_group_totals_reconcile(self):
        """11. Acquisition group totals in summary reconcile with independent acquisition counts."""
        df_acq = pd.read_csv(LANGTANG_ACQ_CSV)
        assert len(df_acq) == 68, f"Expected 68 independent acquisitions, found {len(df_acq)}"
        assert df_acq["product_count"].sum() == 92, "Product counts across acquisitions must sum to 92"

        # Check SAR vs Optical split
        sar_count = len(df_acq[df_acq["sensor_family"] == "SAR"])
        optical_count = len(df_acq[df_acq["sensor_family"] == "OPTICAL"])
        assert sar_count == 23
        assert optical_count == 45

    def test_12_separate_aoi_identifiers_and_coverage(self):
        """12. Langtang and Budhi Gandaki satellite inventories have separate AOI identifiers."""
        df_langtang = pd.read_csv(LANGTANG_SCENES_CSV)
        df_budhi = pd.read_csv(BUDHI_SCENES_CSV)

        lt_aoi = set(df_langtang["aoi_name"].unique())
        bg_aoi = set(df_budhi["aoi_name"].unique())

        assert lt_aoi == {"LANGTANG_LIRUNG_AOI"}
        assert bg_aoi == {"BUDHI_GANDAKI_BASIN"}
        assert lt_aoi.isdisjoint(bg_aoi)

    def test_13_distinction_between_discovery_and_processing(self):
        """13. Distinction between scene discovery and raster processing maintained across records."""
        for csv_path in [LANGTANG_SCENES_CSV, BUDHI_SCENES_CSV]:
            df = pd.read_csv(csv_path)
            for status in df["processing_status"].unique():
                assert "DISCOVERED_IN_CATALOGUE" in status
                assert "PROCESSED" not in status
                assert "ANALYSIS_READY" not in status


class TestModelIsolationAndSecurity:
    """Tests 14-16: ML target isolation, frozen XGBoost SHA-256, and secrets prevention."""

    def test_14_event_records_cannot_integrate_into_glo_lake_targets(self):
        """14. Event records cannot be integrated into GLO lake regression training targets."""
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        for event in data["events"]:
            ml_rel = event["relationship_to_existing_ml"]
            assert ml_rel["is_glo_lake_expansion"] is False
            assert ml_rel["is_in_glo_lake_training_footprint"] is False
            assert ml_rel["predictable_by_fused_xgboost_regressor"] is False
            assert ml_rel["predictable_by_annual_lake_classifier"] is False

        with open(SPATIAL_AUDIT_JSON, "r", encoding="utf-8") as f:
            sp_data = json.load(f)
        feat_rep = sp_data["representation_in_ml_training_features"]
        assert feat_rep["is_source_location_in_training_data"] is False
        assert feat_rep["is_affected_corridor_in_training_data"] is False

    def test_15_frozen_xgboost_sha256_unmodified(self):
        """15. Frozen XGBoost model SHA-256 matches baseline byte-for-byte."""
        assert FROZEN_MODEL_PKL.is_file(), f"Missing frozen model at {FROZEN_MODEL_PKL}"
        h = hashlib.sha256()
        with open(FROZEN_MODEL_PKL, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        current_sha = h.hexdigest()
        assert current_sha == EXPECTED_FROZEN_SHA256, (
            f"Frozen model SHA-256 changed!\nExpected: {EXPECTED_FROZEN_SHA256}\nFound:    {current_sha}"
        )

    def test_16_no_credentials_or_tokens_in_reports(self):
        """16. No raw credentials, tokens, or client secrets appear in any report, CSV, or JSON."""
        # Forbidden tokens or patterns
        forbidden_substrings = [
            "sh-75fe6fab",
            "wyThLEsXUfqapN6BBLhMiqYrXzXskC8n",
            "3de8696c-69ae-482a-a9e3-c9fa946ee12c",
            "client_secret=",
            "Bearer eyJ",
            "CDSE_PASSWORD=",
            "COPERNICUS_PASSWORD="
        ]

        # Scan reports directory and data/events directory
        paths_to_scan = list(REPO_ROOT.glob("reports/**/*.json")) + \
                        list(REPO_ROOT.glob("reports/**/*.csv")) + \
                        list(REPO_ROOT.glob("reports/**/*.md")) + \
                        list(REPO_ROOT.glob("data/events/**/*.json"))

        assert len(paths_to_scan) > 0

        for path in paths_to_scan:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                for forbidden in forbidden_substrings:
                    assert forbidden not in content, (
                        f"Credential or secret found in {path.relative_to(REPO_ROOT)}: {forbidden}"
                    )
            except Exception as e:
                pytest.fail(f"Could not read {path}: {e}")

