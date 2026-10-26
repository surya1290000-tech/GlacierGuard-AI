"""
Dashboard Data Loaders and Schema Validation
=============================================
Reusable, testable data-loading functions for the GlacierGuard-AI
operational demonstration dashboard.  These functions are deliberately
separated from the Streamlit UI so that they can be exercised under
``pytest`` without importing Streamlit.

All file paths are parameterised so that tests can inject synthetic data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── Canonical paths relative to repository root ──────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
QA_CSV_PATH = REPO_ROOT / "reports" / "10_2_five_lake_monitoring_qa.csv"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
MODEL_PATH = REPO_ROOT / "reports" / "models" / "09_final_fused_xgboost.pkl"

# ── Expected schema for the five-lake QA CSV ─────────────────────────────
REQUIRED_QA_COLUMNS = [
    "lake_id",
    "lake_name",
    "glo_2023_vector_ref_km2",
    "latest_s2_date",
    "s2_area_0m_km2",
    "s2_area_50m_km2",
    "latest_s1_date",
    "s1_area_0m_km2",
    "s1_area_50m_km2",
    "temporal_pair_days",
    "temporal_matching_flag",
    "cross_sensor_disagreement_pct",
    "sensor_agreement_status",
    "primary_monitoring_area_km2",
    "diff_from_glo_2023_km2",
    "diff_from_glo_2023_pct",
    "data_quality",
    "monitoring_status",
    "status_rationale",
    "extraction_limitations",
]

# ── Lake metadata (coordinates, identifiers, human-readable names) ───────
LAKE_METADATA: Dict[str, Dict[str, Any]] = {
    "imja_tsho": {
        "display_name": "Imja Tsho",
        "glo_id": "GLO_86.92845_27.89838",
        "lat": 27.89838,
        "lon": 86.92845,
        "utm_epsg": 32645,
        "dam_type": "Moraine-dammed",
        "region": "Khumbu Himal, Nepal",
        "elevation_m": 5010,
        "known_issues": [
            "Monsoon wind roughening elevates VV backscatter in Aug/Sep scenes",
        ],
    },
    "thulagi": {
        "display_name": "Thulagi Glacier Lake",
        "glo_id": "GLO_84.485_28.48833",
        "lat": 28.48833,
        "lon": 84.485,
        "utm_epsg": 32645,
        "dam_type": "End moraine / ice-cored debris",
        "region": "Marsyangdi Basin, Nepal",
        "elevation_m": 4050,
        "known_issues": [
            "Monsoon cloud cover blocks optical monitoring Jun–Sep",
            "135-day temporal gap between S2 and S1 acquisitions",
        ],
    },
    "tsho_rolpa": {
        "display_name": "Tsho Rolpa",
        "glo_id": "GLO_86.47904_27.85858",
        "lat": 27.85858,
        "lon": 86.47904,
        "utm_epsg": 32645,
        "dam_type": "Moraine-dammed (engineered siphon)",
        "region": "Rolwaling Himal, Nepal",
        "elevation_m": 4580,
        "known_issues": [
            "SAR C-band fails in steep Rolwaling gorge (layover/shadow)",
            "Single-sensor optical reliance for monitoring",
        ],
    },
    "lower_barun": {
        "display_name": "Lower Barun",
        "glo_id": "GLO_87.0815_27.84295",
        "lat": 27.84295,
        "lon": 87.0815,
        "utm_epsg": 32645,
        "dam_type": "Moraine-dammed (proglacial)",
        "region": "Makalu-Barun, Nepal",
        "elevation_m": 4550,
        "known_issues": [
            "Proglacial calving retreat exceeds standard 50m buffer",
            "Dynamic adaptive envelope needed for calving bay",
        ],
    },
    "sabai_tsho": {
        "display_name": "Sabai Tsho",
        "glo_id": "GLO_86.62025_27.79158",
        "lat": 27.79158,
        "lon": 86.62025,
        "utm_epsg": 32645,
        "dam_type": "Moraine-dammed (post-breach refilling)",
        "region": "Khumbu Region, Nepal",
        "elevation_m": 4400,
        "known_issues": [
            "Post-1998 breach basin, 1998 baseline only 0.22 km²",
            "Glacial flour / sediment plumes alter reflectance",
            "Shallow moraine margins create partial SAR backscatter",
        ],
    },
}

CANONICAL_LAKE_ORDER = [
    "imja_tsho",
    "thulagi",
    "tsho_rolpa",
    "lower_barun",
    "sabai_tsho",
]


# ── Data loaders ─────────────────────────────────────────────────────────

def load_qa_csv(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and validate the five-lake QA CSV.

    Parameters
    ----------
    csv_path : Path, optional
        Override default path for testing.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame indexed by ``lake_id``.

    Raises
    ------
    FileNotFoundError
        When CSV path does not exist.
    ValueError
        When required columns are missing.
    """
    path = Path(csv_path) if csv_path else QA_CSV_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Five-lake QA CSV not found at: {path}"
        )

    df = pd.read_csv(path)

    # Drop fully empty rows (trailing newlines in CSV)
    df = df.dropna(how="all")

    missing = [c for c in REQUIRED_QA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"QA CSV is missing required columns: {missing}"
        )

    df = df.set_index("lake_id", drop=False)
    return df


def validate_qa_schema(df: pd.DataFrame) -> List[str]:
    """Return a list of schema warnings (empty means all good)."""
    warnings: List[str] = []

    if len(df) == 0:
        warnings.append("QA DataFrame is empty (0 rows)")
        return warnings

    expected_lakes = set(CANONICAL_LAKE_ORDER)
    actual_lakes = set(df["lake_id"].tolist())
    missing_lakes = expected_lakes - actual_lakes
    if missing_lakes:
        warnings.append(f"Missing lakes: {sorted(missing_lakes)}")

    # Check monitoring_status values
    valid_statuses = {"NORMAL", "WATCH", "UNUSUAL_CHANGE", "INSUFFICIENT_DATA"}
    actual_statuses = set(df["monitoring_status"].dropna().unique())
    invalid = actual_statuses - valid_statuses
    if invalid:
        warnings.append(f"Unexpected monitoring_status values: {sorted(invalid)}")

    return warnings


def get_lake_display_name(lake_id: str) -> str:
    """Return human-readable lake name, with fallback."""
    meta = LAKE_METADATA.get(lake_id, {})
    return meta.get("display_name", lake_id.replace("_", " ").title())


def get_lake_qa_record(df: pd.DataFrame, lake_id: str) -> Dict[str, Any]:
    """Extract a single lake's QA record as a dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Full QA DataFrame (from ``load_qa_csv``).
    lake_id : str
        Lake identifier (e.g., ``'imja_tsho'``).

    Returns
    -------
    Dict[str, Any]
        Single-lake metrics dictionary.

    Raises
    ------
    KeyError
        When lake_id is not found in the DataFrame.
    """
    if lake_id not in df.index:
        raise KeyError(f"Lake '{lake_id}' not found in QA data")
    row = df.loc[lake_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row.to_dict()


def compute_staleness_days(
    observation_date_str: Optional[str],
    reference_date: Optional[pd.Timestamp] = None,
) -> Optional[int]:
    """Compute days since last observation.

    Parameters
    ----------
    observation_date_str : str or None
        ISO date string (``'YYYY-MM-DD'``).
    reference_date : pd.Timestamp, optional
        Reference date (defaults to today).

    Returns
    -------
    int or None
        Days since observation, or None if input is null.
    """
    if observation_date_str is None or pd.isna(observation_date_str):
        return None
    obs_date = pd.Timestamp(observation_date_str)
    ref = reference_date or pd.Timestamp.now()
    return max(0, (ref - obs_date).days)


def classify_staleness(days: Optional[int]) -> str:
    """Classify observation staleness into risk tiers.

    Returns one of: ``'CURRENT'``, ``'AGING'``, ``'STALE'``, ``'UNKNOWN'``.
    """
    if days is None:
        return "UNKNOWN"
    if days <= 30:
        return "CURRENT"
    elif days <= 90:
        return "AGING"
    else:
        return "STALE"


def get_water_mask_path(lake_id: str) -> Optional[Path]:
    """Return the path to a lake's water-mask figure, if it exists."""
    fig_name = f"10_2_{lake_id}_water_mask.png"
    fig_path = FIGURES_DIR / fig_name
    return fig_path if fig_path.is_file() else None


def get_available_figures(prefix: str = "10_2_") -> List[Path]:
    """List all figures matching a prefix in the reports/figures directory."""
    if not FIGURES_DIR.is_dir():
        return []
    return sorted(FIGURES_DIR.glob(f"{prefix}*.png"))


# ── Benchmark metrics (frozen; sourced from NB09 validation) ─────────────
FROZEN_BENCHMARK = {
    "model": "S1+S2 Fused XGBoost (NB09)",
    "test_period": "2023 (frozen, non-overlapping)",
    "mae_km2": 0.004792,
    "rmse_km2": 0.015079,
    "r2": 0.3366,
    "training_period": "2017–2021",
    "validation_period": "2022",
    "note": (
        "Benchmark computed on 2023 held-out test set that was never used "
        "during training or hyperparameter tuning. R² of 0.34 reflects "
        "inherent difficulty of predicting year-to-year glacial lake area "
        "changes at sub-0.01 km² scale."
    ),
}


def load_dashboard_data_from_db(
    db_path: Optional[Path] = None,
    csv_fallback_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load latest surveillance monitoring records from SQLite, falling back to CSV.

    Parameters
    ----------
    db_path : Path, optional
        Path to SQLite database file.
    csv_fallback_path : Path, optional
        Fallback CSV path if database is empty or unavailable.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``lake_id``.
    """
    from glacierguard.db import get_latest_monitoring_alerts, init_db

    try:
        init_db(db_path)
        alerts = get_latest_monitoring_alerts(db_path)
        if alerts:
            rows = []
            for lake_id in CANONICAL_LAKE_ORDER:
                if lake_id in alerts:
                    alt = alerts[lake_id]
                    meta = LAKE_METADATA.get(lake_id, {})
                    ev = alt.evidence or {}
                    row = {
                        "lake_id": alt.lake_id,
                        "lake_name": meta.get("display_name", alt.lake_id),
                        "glo_2023_vector_ref_km2": alt.baseline_area_km2,
                        "latest_s2_date": ev.get("s2_date"),
                        "s2_area_0m_km2": ev.get("s2_area_0m_km2"),
                        "s2_area_50m_km2": ev.get("s2_area_km2"),
                        "latest_s1_date": ev.get("s1_date"),
                        "s1_area_0m_km2": ev.get("s1_area_0m_km2"),
                        "s1_area_50m_km2": ev.get("s1_area_km2"),
                        "temporal_pair_days": alt.temporal_pair_days,
                        "temporal_matching_flag": alt.temporal_matching_flag or "UNKNOWN",
                        "cross_sensor_disagreement_pct": alt.cross_sensor_disagreement_pct,
                        "sensor_agreement_status": alt.sensor_agreement_status or "NO_DATA",
                        "primary_monitoring_area_km2": alt.primary_area_km2,
                        "diff_from_glo_2023_km2": alt.area_change_km2,
                        "diff_from_glo_2023_pct": alt.area_change_pct,
                        "data_quality": alt.data_quality,
                        "monitoring_status": alt.status,
                        "user_facing_status": alt.user_facing_status,
                        "status_rationale": alt.rationale,
                        "extraction_limitations": alt.limitations,
                        "staleness_days": alt.staleness_days,
                        "last_updated": alt.timestamp,
                    }
                    rows.append(row)
            if rows:
                df = pd.DataFrame(rows)
                df = df.set_index("lake_id", drop=False)
                return df
    except Exception:
        pass

    # Fall back to QA CSV
    return load_qa_csv(csv_fallback_path)
