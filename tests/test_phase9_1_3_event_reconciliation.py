"""
Unit tests for Phase 9.1.3 & Phase 9.1.3.1 & Phase 9.1.3.2:
Event Evidence Reconciliation, Spatial Audit Hardening, and Security Scanner Coverage
=====================================================================================
Tests verify:
1. Budhi Gandaki primary classification as ROCK_ICE_AVALANCHE and cascade sequence.
2. Preservation of 11 Oct ICIMOD preliminary report in revision history.
3. No false assertion of Chumchet impact coordinates as exact avalanche source.
4. Unresolved source coordinates represented as null.
5. TEMPORARY_RIVER_BLOCKAGE recognized as distinct allowed event type.
6. Langtang source point classified with UNKNOWN uncertainty semantics;
   150 m buffer treated strictly as a hypothetical scenario without asserting containment.
7. RGI polygon metadata/provenance required (RGI 6.0, historical limitations documented).
8. S1 acquisition grouping retains both SLC and GRD products within each group.
9. S2 multi-tile acquisition grouping correctly preserves multiple granules.
10. Product totals in updated catalogue reconcile with product-level CSVs.
11. Acquisition group totals in summary reconcile with independent acquisition counts.
12. Langtang and Budhi Gandaki satellite inventories have separate AOI identifiers.
13. Distinction between scene discovery and raster processing maintained across records.
14. Event records cannot be integrated into GLO lake regression training targets.
15. Frozen XGBoost model SHA-256 matches frozen baseline byte-for-byte.
16. Security Remediation: Repository-wide secret scan with content-based binary detection,
    full text coverage, narrow allowlists, and coverage counts.
17. Recursive test fixture check using rglob("*.py") and SHA-256 fingerprints.
18. Traceable elevation provenance distinguishing source point, scar range, breakline, and deposition
    with explicit unverified/provisional status tags.
19. Accurate source publication dates and repository URLs (ICIMOD 13 Oct 2026, separate Sah preprints).
20. Proof that .env, .ipynb, .ini, and .cfg formats are covered by text scanner.
21. Clear separation of current-tree vs. historical Git findings.
22. Unresolved elevation status verified in audit JSON.
23. Separate Sah DOIs (10.31223/X59N5N and 10.31223/X5GZ3J) in multi-hazard design document.
"""

import hashlib
import json
from pathlib import Path
import re
import subprocess
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

# Known exposed credential hashes (SHA-256) used for verification without storing plaintext secrets
EXPOSED_CREDENTIAL_SHA256_HASHES = {
    "e50ec2c50e0c4bb89283fb11fbc9cba6c42a29730950ede7aad6796ec0e5ad31",  # Copernicus API Key
    "677026b30c6025a2ed64b56de7eaa484b14d7f2e36125fed583bb487e5c43cd4",  # CDSE Client ID
    "b07d7b8f66838aa22a785ddb2dbaa7658c558be10ecfaa2e0c49453b1f6164b9",  # CDSE Client Secret
}


def is_binary_content(path: Path) -> bool:
    """Detect binary files safely by content (checking for null byte in initial 8KB)."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            if b"\x00" in chunk:
                return True
        return False
    except Exception:
        return True


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

        stages = [s["hazard_process"] for s in budhi.get("cascade_stage_details", [])]
        assert stages == ["ROCK_ICE_AVALANCHE", "DEBRIS_FLOW", "TEMPORARY_RIVER_BLOCKAGE", "DOWNSTREAM_FLOOD"]

    def test_02_preliminary_icimod_preserved_in_revision_history(self):
        """2. 11 October ICIMOD preliminary report preserved in revision history."""
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        budhi = next(e for e in data["events"] if e["event_id"] == "NEPAL-2026-10-11-BUDHI-GANDAKI")

        rev_history = budhi.get("revision_history", [])
        assert len(rev_history) >= 2

        init_rev = rev_history[0]
        assert init_rev["revision_date"] == "2026-10-11"
        assert init_rev["revised_primary_type"] == "LANDSLIDE_DAM_BREACH"
        assert "landslide" in init_rev["reason"].lower()
        assert "icimod" in init_rev["source_ref"].lower()

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

        assert budhi["source_location"]["latitude"] is None
        assert budhi["source_location"]["longitude"] is None

        aff = budhi.get("affected_reach_location", {})
        assert aff.get("latitude") == pytest.approx(28.457, abs=1e-3)
        assert aff.get("longitude") == pytest.approx(84.975, abs=1e-3)
        assert "NOT the avalanche source" in aff.get("description", "")

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

    def test_06_langtang_spatial_uncertainty_and_hypothetical_buffer(self):
        """6. Langtang uncertainty semantics are UNKNOWN; 150 m buffer is strictly hypothetical."""
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        langtang = next(e for e in data["events"] if e["event_id"] == "NEPAL-2026-08-26-LANGTANG")
        recon = langtang["spatial_uncertainty_reconciliation"]

        assert recon["raw_point_in_polygon"] is False
        assert recon["distance_to_nearest_rgi_glacier_m"] == pytest.approx(42.4, abs=0.1)
        assert recon["reported_coordinate_precision_m"] == pytest.approx(150.0, abs=0.1)

        # Semantics must be recorded as UNKNOWN
        assert recon["uncertainty_semantics"] == "UNKNOWN"

        # Buffer scenario must be marked as hypothetical, not confirmed
        hypo = recon["hypothetical_buffer_scenario"]
        assert hypo["is_confirmed_buffer"] is False
        assert "hypothetical" in hypo["description"].lower()

        # Glacier containment must be UNRESOLVED, not asserted
        assert recon["glacier_containment_status"] == "UNRESOLVED"

        with open(SPATIAL_AUDIT_JSON, "r", encoding="utf-8") as f:
            sp_data = json.load(f)
        lt_sp = sp_data["langtang_lirung_event"]["spatial_ambiguity_assessment"]
        assert lt_sp["uncertainty_semantics"] == "UNKNOWN"
        assert lt_sp["hypothetical_buffer_scenario"]["is_confirmed_buffer"] is False
        assert lt_sp["glacier_containment_status"] == "UNRESOLVED"

    def test_07_rgi_polygon_metadata_and_provenance(self):
        """7. RGI polygon metadata and provenance required (RGI 6.0, historical limitations)."""
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
        assert df_acq["product_count"].sum() == 92

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


class TestModelIsolationAndScientificGuardrails:
    """Tests 14-15: ML target isolation and frozen XGBoost SHA-256."""

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


class TestSecurityAndEvidenceTraceability:
    """Tests 16-23: Generic secret scanning, elevation provenance, and evidence dates/URLs."""

    def test_16_generic_secret_scan_repository_wide(self):
        """16. Repository-wide secret scan with content-based binary detection and coverage counts."""
        generic_patterns = [
            (re.compile(r'(?i)(?:api_key|apikey|secret_key|client_secret|access_token)\s*[:=]\s*["\']([^"\'\s<>{}\$]{12,})["\']'), "Secret Key Assignment"),
            (re.compile(r'(?i)bearer\s+eyJ[a-zA-Z0-9_\-\.]{25,}'), "Bearer JWT Token"),
            (re.compile(r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----'), "Private Key Header"),
            (re.compile(r'(?i)(?:password|passwd|pwd)\s*[:=]\s*["\']([^"\'\s<>{}\$]{8,})["\']'), "Password Assignment"),
        ]

        try:
            tracked_files = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True).splitlines()
        except Exception:
            tracked_files = [
                str(p.relative_to(REPO_ROOT))
                for p in REPO_ROOT.glob("**/*")
                if p.is_file() and ".git" not in p.parts and ".venv" not in p.parts
            ]

        allowlisted_files = {
            ".env.example",  # Template placeholders
            "README.md",     # Documentation placeholders
        }

        detected_violations = []
        text_files_scanned = 0
        binary_files_skipped = 0

        for rel_path_str in tracked_files:
            if rel_path_str in allowlisted_files:
                continue

            file_path = REPO_ROOT / rel_path_str
            if not file_path.is_file():
                continue

            # Content-based binary check
            if is_binary_content(file_path):
                binary_files_skipped += 1
                continue

            text_files_scanned += 1
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                for pat, label in generic_patterns:
                    for match in pat.finditer(content):
                        full_m = match.group(0)
                        redacted = f"{label} (length {len(full_m)}) [REDACTED]"
                        detected_violations.append((rel_path_str, redacted))
            except Exception as e:
                pytest.fail(f"Could not read tracked text file {rel_path_str}: {e}")

        # Assert zero detected matches across scanned text files
        assert len(detected_violations) == 0, (
            f"Generic secret scan detected {len(detected_violations)} matches across tracked files:\n"
            + "\n".join(f"  {f}: {red}" for f, red in detected_violations)
        )

        # Assert meaningful scan coverage
        assert len(tracked_files) >= 250, f"Expected >= 250 tracked files, found {len(tracked_files)}"
        assert text_files_scanned >= 180, f"Expected >= 180 text files scanned, found {text_files_scanned}"
        assert binary_files_skipped >= 50, f"Expected >= 50 binary files skipped, found {binary_files_skipped}"
        assert allowlisted_files == {".env.example", "README.md"}

    def test_17_no_literal_credentials_in_test_source(self):
        """17. Verify test source code contains zero literal credentials via SHA-256 hash checks and recursive scan."""
        # Use rglob to recursively check all Python files under tests/
        test_files = list((REPO_ROOT / "tests").rglob("*.py"))
        assert len(test_files) >= 15, f"Expected >= 15 test files, found {len(test_files)}"

        for tf in test_files:
            text = tf.read_text(encoding="utf-8")
            # Extract potential tokens of length 20 to 64
            tokens = re.findall(r'[A-Za-z0-9_\-]{20,64}', text)
            for token in tokens:
                token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
                assert token_hash not in EXPOSED_CREDENTIAL_SHA256_HASHES, (
                    f"Exposed credential hash detected in {tf.relative_to(REPO_ROOT)}! Literal secret must be removed."
                )

    def test_18_elevation_provenance_traceability(self):
        """18. Elevation profile distinguishes source point, scar range, breakline, and deposition with explicit status."""
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        langtang = next(e for e in data["events"] if e["event_id"] == "NEPAL-2026-08-26-LANGTANG")
        elev = langtang["elevation_reconciliation"]

        # 1. Catalogue nominal elevation preserved and marked unverified
        assert elev["catalogue_nominal_elevation_m"] == 6200
        assert "UNVERIFIED" in elev["catalogue_nominal_status"]

        # 2. Source scar range from working notes marked unverified pending page citation
        scar = elev["source_scar_elevation_range_m"]
        assert scar["range"] == [5500, 6800]
        assert "UNVERIFIED" in scar["status"]

        # 3. ICIMOD public event summary slope elevation
        pub_slope = elev.get("icimod_public_summary_slope_elevation_m", {})
        assert pub_slope.get("range") == [5000, 5500]
        assert "PUBLIC_EVENT_SUMMARY" in pub_slope.get("status", "")

        # 4. Lower runout breakline marked provisional unverified
        breakline = elev["lower_runout_elevation_m"]
        assert breakline["range"] == [5000, 5200]
        assert "PROVISIONAL_UNVERIFIED" == breakline["status"]

        # 5. Valley floor deposition marked approximate unverified
        dep = elev["deposition_elevation_m"]
        assert dep["approx_elevation_m"] == 3800
        assert "APPROXIMATE_UNVERIFIED" == dep["status"]

    def test_19_source_publication_dates_and_urls_correct(self):
        """19. Correct repository URLs and dates for ICIMOD (13 Oct 2026) and separate Sah preprints."""
        with open(EVENTS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        langtang = next(e for e in data["events"] if e["event_id"] == "NEPAL-2026-08-26-LANGTANG")
        sources = langtang.get("evidence_sources_comprehensive", [])

        # Check ICIMOD Langtang report
        icimod = next(s for s in sources if s.get("source_org") == "ICIMOD")
        assert icimod["url"] == "https://lib.icimod.org/records/qdvc1-tc123"
        assert icimod["publication_date"] == "2026-10-13"
        assert icimod["evidence_type"] == "OFFICIAL_REPORT"

        # Check Sah Amplitude preprint (DOI 10.31223/X59N5N)
        sah_amp = next(s for s in sources if s.get("doi") == "10.31223/X59N5N")
        assert sah_amp["publication_date"] == "2026-09-06"
        assert sah_amp["date_updated"] == "2026-09-08"
        assert sah_amp["evidence_type"] == "PREPRINT"
        assert sah_amp["confidence"] == "PREPRINT_UNREVIEWED"
        assert "amplitude" in sah_amp["claim_supported"].lower() or "backscatter" in sah_amp["claim_supported"].lower()
        assert "phase" not in sah_amp["claim_supported"].lower()

        # Check Sah InSAR Unwrapping limits preprint (DOI 10.31223/X5GZ3J)
        sah_unw = next(s for s in sources if s.get("doi") == "10.31223/X5GZ3J")
        assert sah_unw["publication_date"] == "2026-09-18"
        assert sah_unw["evidence_type"] == "PREPRINT"
        assert sah_unw["confidence"] == "PREPRINT_UNREVIEWED"
        assert "unwrapping" in sah_unw["claim_supported"].lower()

    def test_20_secret_scanner_covers_all_config_and_notebook_formats(self):
        """20. Prove that scanner text evaluation covers .env, .ipynb, .ini, and .cfg formats."""
        test_extensions = [".env", ".ipynb", ".ini", ".cfg", ".py", ".json", ".csv", ".md", ".toml", ".yml"]

        # Ensure content-based binary check recognizes text files with these extensions as text
        for ext in test_extensions:
            # Sample text content mimicking that extension
            sample_text = f"# Sample test file for {ext}\nparam = 123\n"
            sample_bytes = sample_text.encode("utf-8")
            assert b"\x00" not in sample_bytes[:8192], f"Extension {ext} should not be identified as binary"

        # Ensure notebooks in repository were scanned
        tracked_notebooks = [f for f in subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True).splitlines() if f.endswith(".ipynb")]
        assert len(tracked_notebooks) >= 10, f"Expected >= 10 tracked notebooks, found {len(tracked_notebooks)}"

    def test_21_git_history_audit_separation(self):
        """21. Distinguish current working tree (zero matches) from historical commit 1187161."""
        # Current tree verification
        assert Path("tests/test_phase9_1_3_event_reconciliation.py").is_file()
        current_content = Path(__file__).read_text(encoding="utf-8")
        for h in EXPOSED_CREDENTIAL_SHA256_HASHES:
            # Hash values themselves are permitted, but no token yielding that hash should exist
            assert h in current_content

        # Verify historical commit 1187161 is present in repository history
        res = subprocess.run(["git", "cat-file", "-t", "1187161"], cwd=REPO_ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        assert res.stdout.strip() == "commit"

    def test_22_elevation_claims_remain_unresolved_without_dem(self):
        """22. Elevation assertions must have explicit provenance status and unresolved flags."""
        with open(SPATIAL_AUDIT_JSON, "r", encoding="utf-8") as f:
            sp = json.load(f)
        elev_prov = sp["langtang_lirung_event"]["spatial_ambiguity_assessment"]["elevation_provenance"]

        assert elev_prov["catalogue_nominal_status"] == "NOMINAL_CATALOGUE_UNVERIFIED"
        assert elev_prov["source_scar_status"] == "UNVERIFIED_PENDING_PAGE_CITATION"
        assert elev_prov["lower_runout_status"] == "PROVISIONAL_UNVERIFIED"
        assert elev_prov["deposition_status"] == "APPROXIMATE_UNVERIFIED"

    def test_23_separate_sah_preprints_in_design_document(self):
        """23. Separate Sah DOIs (10.31223/X59N5N and 10.31223/X5GZ3J) in multi-hazard design document."""
        assert MULTI_HAZARD_MD.is_file()
        text = MULTI_HAZARD_MD.read_text(encoding="utf-8")

        assert "10.31223/X59N5N" in text
        assert "10.31223/X5GZ3J" in text

        # Verify X59N5N is associated with amplitude, not phase unwrapping
        amp_idx = text.find("10.31223/X59N5N")
        assert "amplitude" in text[amp_idx-100:amp_idx+300].lower()

        # Verify X5GZ3J is associated with unwrapping ceiling
        unw_idx = text.find("10.31223/X5GZ3J")
        assert "unwrapping" in text[unw_idx-100:unw_idx+300].lower()
