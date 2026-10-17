"""
GlacierGuard-AI — Unit test suite for HMAGLOFDB audit and data quality framework.

Tests cover:
  1. Placeholder and whitespace normalization
  2. '+' representing a positive-but-unknown count
  3. Duplicate event identifiers
  4. Missing or malformed event years
  5. Empty or missing input files
  6. Correct classification of target, identifier, evaluation-only, and post-event leakage fields
  7. Reproducible profile generation on a tiny synthetic fixture
"""

import sys
import pathlib
import pytest
import pandas as pd
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scratch"))

from audit_glof_event_database import (
    normalize_value,
    parse_plus_count_field,
    load_and_clean_hmaglofdb,
    audit_hmaglofdb,
    COLUMN_REGISTRY,
    PLUS_COUNT_FIELDS,
    NA_PLACEHOLDERS
)


# -----------------------------------------------------------------------------
# Category 1: Placeholder and Whitespace Normalization
# -----------------------------------------------------------------------------
class TestNormalization:
    @pytest.mark.parametrize("raw_input", ["", " ", "\xa0", "\t", "  \xa0  "])
    def test_whitespace_and_nbsp_stripped(self, raw_input):
        assert normalize_value(raw_input) is None

    @pytest.mark.parametrize("placeholder", ["nan", "NaN", "None", "null", "-", "--", "?", "Unknown", "Na"])
    def test_na_placeholders_recognized(self, placeholder):
        assert normalize_value(placeholder) is None

    def test_valid_text_preserved(self):
        assert normalize_value("  Imja Tsho  ") == "Imja Tsho"
        assert normalize_value("Moraine-dammed\xa0lake") == "Moraine-dammed lake"


# -----------------------------------------------------------------------------
# Category 2: '+' Representing Positive-but-Unknown Count
# -----------------------------------------------------------------------------
class TestPlusCounts:
    def test_standalone_plus_is_unknown(self):
        status, val = parse_plus_count_field("+")
        assert status == "PLUS_UNKNOWN"
        assert val is None

    def test_plus_with_lower_bound(self):
        status, val = parse_plus_count_field("15+")
        assert status == "PLUS_LOWER_BOUND"
        assert val == 15.0

    def test_exact_numeric(self):
        status, val = parse_plus_count_field("42")
        assert status == "EXACT_NUMERIC"
        assert val == 42.0

        status, val = parse_plus_count_field("1,250")
        assert status == "EXACT_NUMERIC"
        assert val == 1250.0

    def test_plus_never_converted_to_zero(self):
        status, val = parse_plus_count_field("+")
        assert val != 0
        assert val is None


# -----------------------------------------------------------------------------
# Category 3: Duplicate Event Identifiers
# -----------------------------------------------------------------------------
class TestIdentifierAudit:
    def test_detects_duplicate_and_null_ids(self, tmp_path):
        csv_file = tmp_path / "dup_test.csv"
        df = pd.DataFrame({
            "GF_ID": ["GF001", "GF002", "GF002", None, "GF003"],
            "Year_exact": [2001, 2002, 2003, 2004, 2005],
            "Lat_lake": [27.9, 28.0, 28.1, 28.2, 28.3]
        })
        df.to_csv(csv_file, index=False)

        profile = audit_hmaglofdb(csv_file, tmp_path / "out")
        id_audit = profile["identifier_audit"]
        assert id_audit["unique_identifiers"] == 3
        assert id_audit["duplicate_identifiers"] == 1
        assert id_audit["null_identifiers"] == 1


# -----------------------------------------------------------------------------
# Category 4: Missing or Malformed Event Years
# -----------------------------------------------------------------------------
class TestEventYears:
    def test_malformed_and_missing_years(self, tmp_path):
        csv_file = tmp_path / "years_test.csv"
        df = pd.DataFrame({
            "GF_ID": ["GF01", "GF02", "GF03", "GF04"],
            "Year_exact": ["2016", "circa 1985", None, "1998.0"]
        })
        df.to_csv(csv_file, index=False)

        profile = audit_hmaglofdb(csv_file, tmp_path / "out")
        temporal = profile["temporal_coverage"]
        assert temporal["earliest_year"] == 1998
        assert temporal["latest_year"] == 2016
        assert temporal["malformed_year_count"] == 1


# -----------------------------------------------------------------------------
# Category 5: Empty or Missing Input Files
# -----------------------------------------------------------------------------
class TestMissingFiles:
    def test_missing_file_raises_actionable_error(self, tmp_path):
        non_existent = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError) as exc_info:
            load_and_clean_hmaglofdb(non_existent)
        assert "HMAGLOFDB CSV file not found" in str(exc_info.value)
        assert "Please download the dataset" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Category 6: Column Classification & Leakage Audit
# -----------------------------------------------------------------------------
class TestLeakageClassification:
    def test_post_event_leakage_fields_correctly_flagged(self):
        post_event_fields = [
            "Impact", "Impact_type", "Driver_GLOF", "Mechanism",
            "Discharge_water", "Lives_total", "Infra", "Hydropower",
            "Econ_damage", "Sat_evidence"
        ]
        for f in post_event_fields:
            assert f in COLUMN_REGISTRY, f"Field {f} missing from registry"
            role, _ = COLUMN_REGISTRY[f]
            assert role == "POST_EVENT_LEAKAGE", f"Field {f} must be POST_EVENT_LEAKAGE, got {role}"

    def test_pre_event_candidate_fields(self):
        pre_event_fields = ["Lat_lake", "Lon_lake", "Elev_lake", "Lake_type"]
        for f in pre_event_fields:
            assert f in COLUMN_REGISTRY, f"Field {f} missing from registry"
            role, _ = COLUMN_REGISTRY[f]
            assert role == "PRE_EVENT_CANDIDATE", f"Field {f} must be PRE_EVENT_CANDIDATE, got {role}"

    def test_identifiers_flagged(self):
        for f in ["GF_ID", "Lake_name", "Glacier_name"]:
            role, _ = COLUMN_REGISTRY[f]
            assert role == "IDENTIFIER"


# -----------------------------------------------------------------------------
# Category 7: Reproducible Profile Generation on Synthetic Fixture
# -----------------------------------------------------------------------------
class TestProfileGeneration:
    def test_profile_generation_synthetic(self, tmp_path):
        fixture_csv = tmp_path / "synthetic_fixture.csv"
        out_dir = tmp_path / "fixture_reports"

        df_synth = pd.DataFrame({
            "GF_ID": ["GF01", "GF02", "GF03"],
            "Year_exact": ["2015", "2018", "2021"],
            "Lat_lake": ["27.9", "28.1", "27.8"],
            "Lon_lake": ["86.9", "85.4", "86.5"],
            "Elev_lake": ["5010", "4800", "4500"],
            "Lake_type": ["Moraine-dammed", "Ice-dammed", "Moraine-dammed"],
            "Lives_total": ["+", "0", "12"],
            "Impact": [None, "Damaged village bridge 10km downstream", None]
        })
        df_synth.to_csv(fixture_csv, index=False)

        profile = audit_hmaglofdb(fixture_csv, out_dir)
        assert profile["dimensions"]["total_records"] == 3
        assert profile["identifier_audit"]["unique_identifiers"] == 3
        assert profile["plus_counts_audit"]["fields_with_plus_counts"].get("Lives_total") == 1
        assert (out_dir / "hmaglofdb_dataset_profile.json").exists()
        assert (out_dir / "hmaglofdb_field_completeness.csv").exists()
        assert (out_dir / "hmaglofdb_feature_registry.csv").exists()
        assert (out_dir / "hmaglofdb_temporal_qc_audit.csv").exists()

    def test_post_2025_cross_check_flagging(self, tmp_path):
        """Events dated after 2025-12-31 must receive explicit cross-check flag."""
        fixture_csv = tmp_path / "post_2025_fixture.csv"
        out_dir = tmp_path / "qc_reports"

        df_synth = pd.DataFrame({
            "GF_ID": ["GF_OLD", "GF_MODERN", "GF_FUTURE"],
            "Year_exact": ["1780", "2021", "2026"],
            "Month": ["6", "7", "8"],
            "Day": ["1", "15", "9"],
            "Lake_name": ["Ancient", "Current", "Provisional"],
            "Country": ["India", "Nepal", "Bhutan"],
            "Ref_scientific": ["(Strachey, 1853)", "(This study)", "(This study)"]
        })
        df_synth.to_csv(fixture_csv, index=False)

        profile = audit_hmaglofdb(fixture_csv, out_dir)
        df_qc = pd.read_csv(out_dir / "hmaglofdb_temporal_qc_audit.csv")

        # Row 0: 1780 -> historical pre-instrumental
        assert df_qc.loc[0, "QC_Flag"] == "HISTORICAL_PRE_INSTRUMENTAL_DOCUMENTARY"
        # Row 1: 2021 -> modern satellite era
        assert df_qc.loc[1, "QC_Flag"] == "MODERN_SATELLITE_ERA"
        # Row 2: 2026 -> post-2025 unverified cross check required
        assert df_qc.loc[2, "QC_Flag"] == "UNVERIFIED_POST_2025_REQUIRES_CROSS_CHECK"
        assert df_qc.loc[2, "Review_Status"] == "PENDING_MAINTAINER_EVALUATION"
        assert "maintainer_documentation_notice" in profile["temporal_qc_audit"]
