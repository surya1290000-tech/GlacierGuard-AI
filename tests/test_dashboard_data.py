"""
Tests for Phase 6 Dashboard Data Loaders
==========================================
Validates data loading, schema validation, lake selection, QA status
handling, staleness calculation, and missing/empty report handling.
"""

import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from glacierguard.dashboard_data import (
    CANONICAL_LAKE_ORDER,
    FROZEN_BENCHMARK,
    LAKE_METADATA,
    REQUIRED_QA_COLUMNS,
    classify_staleness,
    compute_staleness_days,
    get_available_figures,
    get_lake_display_name,
    get_lake_qa_record,
    get_water_mask_path,
    load_qa_csv,
    validate_qa_schema,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_qa_csv(tmp_path: Path) -> Path:
    """Create a minimal valid QA CSV for testing."""
    csv_path = tmp_path / "test_qa.csv"
    header = ",".join(REQUIRED_QA_COLUMNS) + "\n"
    row = (
        "imja_tsho,Imja Tsho,1.7646,2026-09-28,1.765,2.15,"
        "2026-09-28,1.368,1.633,0,SAME_DAY (0d),27.3,"
        "MODERATE_DISAGREEMENT,1.765,0.0005,0.03,HIGH,NORMAL,"
        "Stable extent,Wind roughening\n"
    )
    csv_path.write_text(header + row)
    return csv_path


@pytest.fixture
def full_qa_csv(tmp_path: Path) -> Path:
    """Create a QA CSV with all five lakes."""
    csv_path = tmp_path / "full_qa.csv"
    header = ",".join(REQUIRED_QA_COLUMNS)
    rows = [
        header,
        "imja_tsho,Imja Tsho,1.7646,2026-09-28,1.765,2.15,2026-09-28,1.368,1.633,0,SAME_DAY (0d),27.3,MODERATE_DISAGREEMENT,1.765,0.0005,0.03,HIGH,NORMAL,Stable extent,Wind roughening",
        "thulagi,Thulagi Glacier Lake,1.0155,2026-05-14,0.953,0.993,2026-09-26,0.471,0.639,135,TEMPORALLY_UNMATCHED (135d),,TEMPORALLY_UNMATCHED,0.993,-0.022,-2.17,MEDIUM,WATCH,Monsoon gap,Cloud cover",
        "tsho_rolpa,Tsho Rolpa,1.6632,2026-09-28,1.654,1.735,2026-09-28,0.053,0.083,0,SAME_DAY (0d),,SAR_UNRELIABLE_CURRENT_SETTINGS,1.735,0.072,4.32,MEDIUM,WATCH,SAR unreliable,Gorge layover",
        "lower_barun,Lower Barun,0.4194,2026-09-28,0.419,0.609,2026-09-28,0.368,0.532,0,SAME_DAY (0d),13.6,GOOD_AGREEMENT,0.609,0.190,45.21,HIGH,UNUSUAL_CHANGE,Calving expansion,Buffer exceeded",
        "sabai_tsho,Sabai Tsho,0.4557,2026-09-28,0.259,0.518,2026-09-28,0.233,0.381,0,SAME_DAY (0d),30.3,MODERATE_DISAGREEMENT,0.518,0.062,13.58,MEDIUM,WATCH,Gradual refilling,Sediment plumes",
    ]
    csv_path.write_text("\n".join(rows) + "\n")
    return csv_path


@pytest.fixture
def empty_qa_csv(tmp_path: Path) -> Path:
    """Create a CSV with header but no data rows."""
    csv_path = tmp_path / "empty_qa.csv"
    header = ",".join(REQUIRED_QA_COLUMNS) + "\n"
    csv_path.write_text(header)
    return csv_path


# ─────────────────────────────────────────────────────────────────────────
# Data Loading Tests
# ─────────────────────────────────────────────────────────────────────────

class TestLoadQaCsv:
    """Tests for load_qa_csv()."""

    def test_load_valid_csv(self, minimal_qa_csv: Path):
        """Loading a valid CSV returns a non-empty DataFrame."""
        df = load_qa_csv(minimal_qa_csv)
        assert len(df) == 1
        assert "imja_tsho" in df.index

    def test_load_full_csv_all_lakes(self, full_qa_csv: Path):
        """Loading a full CSV returns all five lakes."""
        df = load_qa_csv(full_qa_csv)
        assert len(df) == 5
        for lake_id in CANONICAL_LAKE_ORDER:
            assert lake_id in df.index

    def test_load_missing_file_raises(self, tmp_path: Path):
        """Loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_qa_csv(tmp_path / "nonexistent.csv")

    def test_load_missing_columns_raises(self, tmp_path: Path):
        """Loading a CSV with missing required columns raises ValueError."""
        csv_path = tmp_path / "bad_schema.csv"
        csv_path.write_text("lake_id,lake_name\nimja_tsho,Imja Tsho\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_qa_csv(csv_path)

    def test_load_empty_csv_returns_empty(self, empty_qa_csv: Path):
        """Loading a CSV with only headers returns an empty DataFrame."""
        df = load_qa_csv(empty_qa_csv)
        assert len(df) == 0

    def test_trailing_empty_rows_dropped(self, tmp_path: Path):
        """Trailing empty rows in the CSV are dropped."""
        csv_path = tmp_path / "trailing.csv"
        header = ",".join(REQUIRED_QA_COLUMNS)
        row = (
            "imja_tsho,Imja Tsho,1.7646,2026-09-28,1.765,2.15,"
            "2026-09-28,1.368,1.633,0,SAME_DAY (0d),27.3,"
            "MODERATE_DISAGREEMENT,1.765,0.0005,0.03,HIGH,NORMAL,"
            "Stable,Wind"
        )
        csv_path.write_text(f"{header}\n{row}\n\n\n")
        df = load_qa_csv(csv_path)
        assert len(df) == 1


# ─────────────────────────────────────────────────────────────────────────
# Schema Validation Tests
# ─────────────────────────────────────────────────────────────────────────

class TestValidateQaSchema:
    """Tests for validate_qa_schema()."""

    def test_valid_full_schema_no_warnings(self, full_qa_csv: Path):
        """Full CSV with all five lakes produces no schema warnings."""
        df = load_qa_csv(full_qa_csv)
        warnings = validate_qa_schema(df)
        assert warnings == []

    def test_missing_lake_warning(self, minimal_qa_csv: Path):
        """CSV missing lakes produces a warning about missing lakes."""
        df = load_qa_csv(minimal_qa_csv)
        warnings = validate_qa_schema(df)
        assert any("Missing lakes" in w for w in warnings)

    def test_empty_dataframe_warning(self, empty_qa_csv: Path):
        """Empty DataFrame produces an empty-rows warning."""
        df = load_qa_csv(empty_qa_csv)
        warnings = validate_qa_schema(df)
        assert any("empty" in w.lower() for w in warnings)

    def test_invalid_status_warning(self, tmp_path: Path):
        """Invalid monitoring_status values produce a warning."""
        csv_path = tmp_path / "bad_status.csv"
        header = ",".join(REQUIRED_QA_COLUMNS)
        row = (
            "imja_tsho,Imja Tsho,1.7646,2026-09-28,1.765,2.15,"
            "2026-09-28,1.368,1.633,0,SAME_DAY (0d),27.3,"
            "MODERATE_DISAGREEMENT,1.765,0.0005,0.03,HIGH,INVALID_STATUS,"
            "Stable,Wind"
        )
        csv_path.write_text(f"{header}\n{row}\n")
        df = load_qa_csv(csv_path)
        warnings = validate_qa_schema(df)
        assert any("Unexpected" in w for w in warnings)


# ─────────────────────────────────────────────────────────────────────────
# Lake Selection Tests
# ─────────────────────────────────────────────────────────────────────────

class TestLakeSelection:
    """Tests for lake selection and record retrieval."""

    def test_get_lake_display_name_known(self):
        """Known lake IDs return their display name."""
        assert get_lake_display_name("imja_tsho") == "Imja Tsho"
        assert get_lake_display_name("lower_barun") == "Lower Barun"

    def test_get_lake_display_name_unknown(self):
        """Unknown lake IDs return a title-cased fallback."""
        assert get_lake_display_name("mystery_lake") == "Mystery Lake"

    def test_get_lake_qa_record(self, full_qa_csv: Path):
        """Retrieving a lake QA record returns the correct data."""
        df = load_qa_csv(full_qa_csv)
        record = get_lake_qa_record(df, "imja_tsho")
        assert record["lake_name"] == "Imja Tsho"
        assert record["monitoring_status"] == "NORMAL"

    def test_get_lake_qa_record_missing_raises(self, full_qa_csv: Path):
        """Requesting a non-existent lake raises KeyError."""
        df = load_qa_csv(full_qa_csv)
        with pytest.raises(KeyError, match="mystery_lake"):
            get_lake_qa_record(df, "mystery_lake")

    def test_all_canonical_lakes_have_metadata(self):
        """All canonical lakes have entries in LAKE_METADATA."""
        for lake_id in CANONICAL_LAKE_ORDER:
            assert lake_id in LAKE_METADATA
            meta = LAKE_METADATA[lake_id]
            assert "display_name" in meta
            assert "lat" in meta
            assert "lon" in meta


# ─────────────────────────────────────────────────────────────────────────
# Staleness Tests
# ─────────────────────────────────────────────────────────────────────────

class TestStaleness:
    """Tests for observation staleness computation."""

    def test_staleness_recent(self):
        """An observation from today has 0 staleness days."""
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        days = compute_staleness_days(today)
        assert days == 0

    def test_staleness_known_date(self):
        """Staleness against a known reference date is computed correctly."""
        days = compute_staleness_days(
            "2026-09-01",
            reference_date=pd.Timestamp("2026-10-01"),
        )
        assert days == 30

    def test_staleness_none_input(self):
        """None input returns None."""
        assert compute_staleness_days(None) is None

    def test_staleness_nan_input(self):
        """NaN input returns None."""
        assert compute_staleness_days(float("nan")) is None

    def test_classify_current(self):
        assert classify_staleness(0) == "CURRENT"
        assert classify_staleness(15) == "CURRENT"
        assert classify_staleness(30) == "CURRENT"

    def test_classify_aging(self):
        assert classify_staleness(31) == "AGING"
        assert classify_staleness(60) == "AGING"
        assert classify_staleness(90) == "AGING"

    def test_classify_stale(self):
        assert classify_staleness(91) == "STALE"
        assert classify_staleness(365) == "STALE"

    def test_classify_unknown(self):
        assert classify_staleness(None) == "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────
# QA Status Tests
# ─────────────────────────────────────────────────────────────────────────

class TestQaStatuses:
    """Tests for QA status values in loaded data."""

    def test_imja_tsho_normal(self, full_qa_csv: Path):
        """Imja Tsho should have NORMAL status."""
        df = load_qa_csv(full_qa_csv)
        rec = get_lake_qa_record(df, "imja_tsho")
        assert rec["monitoring_status"] == "NORMAL"
        assert rec["data_quality"] == "HIGH"

    def test_thulagi_watch(self, full_qa_csv: Path):
        """Thulagi should have WATCH status."""
        df = load_qa_csv(full_qa_csv)
        rec = get_lake_qa_record(df, "thulagi")
        assert rec["monitoring_status"] == "WATCH"

    def test_lower_barun_unusual(self, full_qa_csv: Path):
        """Lower Barun should have UNUSUAL_CHANGE status."""
        df = load_qa_csv(full_qa_csv)
        rec = get_lake_qa_record(df, "lower_barun")
        assert rec["monitoring_status"] == "UNUSUAL_CHANGE"
        assert rec["data_quality"] == "HIGH"

    def test_all_statuses_valid(self, full_qa_csv: Path):
        """All lake statuses in the CSV are valid values."""
        valid = {"NORMAL", "WATCH", "UNUSUAL_CHANGE", "INSUFFICIENT_DATA"}
        df = load_qa_csv(full_qa_csv)
        for lake_id in CANONICAL_LAKE_ORDER:
            rec = get_lake_qa_record(df, lake_id)
            assert rec["monitoring_status"] in valid, (
                f"{lake_id} has invalid status: {rec['monitoring_status']}"
            )


# ─────────────────────────────────────────────────────────────────────────
# Frozen Benchmark Tests
# ─────────────────────────────────────────────────────────────────────────

class TestFrozenBenchmark:
    """Tests for frozen benchmark constants."""

    def test_mae_value(self):
        """Frozen MAE matches NB09 validated value."""
        assert FROZEN_BENCHMARK["mae_km2"] == pytest.approx(0.004792, abs=1e-6)

    def test_rmse_value(self):
        """Frozen RMSE matches NB09 validated value."""
        assert FROZEN_BENCHMARK["rmse_km2"] == pytest.approx(0.015079, abs=1e-6)

    def test_r2_value(self):
        """Frozen R² matches NB09 validated value."""
        assert FROZEN_BENCHMARK["r2"] == pytest.approx(0.3366, abs=1e-4)

    def test_test_period(self):
        """Test period is the frozen 2023 set."""
        assert "2023" in FROZEN_BENCHMARK["test_period"]


# ─────────────────────────────────────────────────────────────────────────
# Measured vs Predicted Separation
# ─────────────────────────────────────────────────────────────────────────

class TestMeasuredVsPredicted:
    """Ensure dashboard correctly distinguishes measured and predicted data."""

    def test_qa_csv_contains_only_measured(self, full_qa_csv: Path):
        """QA CSV rows represent measured satellite observations, not model predictions."""
        df = load_qa_csv(full_qa_csv)
        # The QA CSV contains s2_area and s1_area columns — these are measured
        # There should be no column named 'predicted_area' or similar
        predicted_cols = [c for c in df.columns if "predict" in c.lower()]
        assert predicted_cols == [], (
            f"QA CSV should not contain prediction columns: {predicted_cols}"
        )

    def test_frozen_benchmark_is_separate(self):
        """Benchmark metrics are stored separately from observation data."""
        # Benchmark is a static dict, not derived from the QA CSV
        assert isinstance(FROZEN_BENCHMARK, dict)
        assert "mae_km2" in FROZEN_BENCHMARK
        assert "note" in FROZEN_BENCHMARK


# ─────────────────────────────────────────────────────────────────────────
# Water Mask Figure Tests
# ─────────────────────────────────────────────────────────────────────────

class TestWaterMaskFigures:
    """Tests for water mask figure discovery."""

    def test_known_lake_mask_paths(self):
        """Water mask paths for known lakes are plausible (may or may not exist)."""
        for lake_id in CANONICAL_LAKE_ORDER:
            # We just test the function doesn't crash
            result = get_water_mask_path(lake_id)
            # Result is either None (file missing) or a Path
            assert result is None or isinstance(result, Path)

    def test_available_figures_returns_list(self):
        """get_available_figures always returns a list."""
        figs = get_available_figures("10_2_")
        assert isinstance(figs, list)


# ─────────────────────────────────────────────────────────────────────────
# Integration: Real QA CSV (if available)
# ─────────────────────────────────────────────────────────────────────────

class TestRealQaCsv:
    """Integration tests against the real QA CSV (skipped if missing)."""

    REAL_CSV = Path(__file__).resolve().parents[1] / "reports" / "10_2_five_lake_monitoring_qa.csv"

    @pytest.mark.skipif(
        not REAL_CSV.is_file(),
        reason="Real QA CSV not found at expected location",
    )
    def test_load_real_csv(self):
        """Real QA CSV loads without errors."""
        df = load_qa_csv(self.REAL_CSV)
        assert len(df) >= 5

    @pytest.mark.skipif(
        not REAL_CSV.is_file(),
        reason="Real QA CSV not found at expected location",
    )
    def test_real_csv_schema_valid(self):
        """Real QA CSV passes schema validation."""
        df = load_qa_csv(self.REAL_CSV)
        warnings = validate_qa_schema(df)
        assert warnings == [], f"Schema warnings: {warnings}"

    @pytest.mark.skipif(
        not REAL_CSV.is_file(),
        reason="Real QA CSV not found at expected location",
    )
    def test_real_csv_imja_tsho_normal(self):
        """Real Imja Tsho status is NORMAL."""
        df = load_qa_csv(self.REAL_CSV)
        rec = get_lake_qa_record(df, "imja_tsho")
        assert rec["monitoring_status"] == "NORMAL"

    @pytest.mark.skipif(
        not REAL_CSV.is_file(),
        reason="Real QA CSV not found at expected location",
    )
    def test_real_csv_lower_barun_unusual_change(self):
        """Real Lower Barun status is UNUSUAL_CHANGE."""
        df = load_qa_csv(self.REAL_CSV)
        rec = get_lake_qa_record(df, "lower_barun")
        assert rec["monitoring_status"] == "UNUSUAL_CHANGE"
