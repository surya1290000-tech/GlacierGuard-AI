"""
GlacierGuard-AI SQLite Persistence Layer
========================================
Thread-safe, relational SQLite database layer storing satellite products,
processed lake observations, pipeline execution traces, and operational
risk alerts.
"""

from contextlib import contextmanager
import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional

from glacierguard.config import DEFAULT_DB_PATH, default_settings
from glacierguard.models import (
    MonitoringAlert,
    ObservationRecord,
    PipelineRun,
    SatelliteProduct,
)


DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS satellite_products (
        product_id TEXT PRIMARY KEY,
        collection TEXT NOT NULL,
        acquisition_date TEXT NOT NULL,
        bbox_json TEXT,
        lake_id TEXT,
        cloud_cover_pct REAL,
        platform TEXT,
        orbit_direction TEXT,
        discovered_at TEXT NOT NULL,
        metadata_json TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS observations (
        observation_id TEXT PRIMARY KEY,
        lake_id TEXT NOT NULL,
        sensor TEXT NOT NULL,
        acquisition_date TEXT NOT NULL,
        product_id TEXT,
        area_0m_km2 REAL,
        area_50m_km2 REAL,
        valid_frac REAL,
        cloud_frac REAL,
        vv_median_db REAL,
        created_at TEXT NOT NULL,
        notes TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id TEXT PRIMARY KEY,
        lake_id TEXT NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT,
        stage TEXT NOT NULL,
        status TEXT NOT NULL,
        input_products_json TEXT,
        model_features_json TEXT,
        model_prediction REAL,
        error_message TEXT,
        metadata_json TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS monitoring_alerts (
        alert_id TEXT PRIMARY KEY,
        lake_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        status TEXT NOT NULL,
        user_facing_status TEXT NOT NULL,
        primary_area_km2 REAL,
        baseline_area_km2 REAL,
        area_change_km2 REAL,
        area_change_pct REAL,
        sensor_agreement_status TEXT,
        cross_sensor_disagreement_pct REAL,
        temporal_pair_days INTEGER,
        temporal_matching_flag TEXT,
        data_quality TEXT,
        staleness_days INTEGER,
        evidence_json TEXT,
        rationale TEXT,
        limitations TEXT
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_products_lake_date
    ON satellite_products (lake_id, acquisition_date DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_observations_lake_date
    ON observations (lake_id, acquisition_date DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pipeline_runs_lake_time
    ON pipeline_runs (lake_id, start_time DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_alerts_lake_time
    ON monitoring_alerts (lake_id, timestamp DESC);
    """,
]


@contextmanager
def get_db_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite database connection."""
    target_path = Path(db_path or default_settings.db_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables and indexes."""
    with get_db_connection(db_path) as conn:
        with conn:
            for stmt in DDL_STATEMENTS:
                conn.execute(stmt)


# ── Satellite Products ───────────────────────────────────────────────────────

def save_satellite_products(
    products: List[SatelliteProduct],
    db_path: Optional[Path] = None,
) -> int:
    """Insert or ignore discovered satellite products. Returns number inserted."""
    if not products:
        return 0
    init_db(db_path)
    count = 0
    with get_db_connection(db_path) as conn:
        with conn:
            for p in products:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO satellite_products (
                        product_id, collection, acquisition_date, bbox_json,
                        lake_id, cloud_cover_pct, platform, orbit_direction,
                        discovered_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        p.product_id,
                        p.collection,
                        p.acquisition_date,
                        json.dumps(p.bbox),
                        p.lake_id,
                        p.cloud_cover_pct,
                        p.platform,
                        p.orbit_direction,
                        p.discovered_at or datetime.datetime.utcnow().isoformat() + "Z",
                        json.dumps(p.metadata),
                    ),
                )
                if cur.rowcount > 0:
                    count += 1
    return count


def get_satellite_products(
    lake_id: Optional[str] = None,
    limit: int = 50,
    db_path: Optional[Path] = None,
) -> List[SatelliteProduct]:
    """Retrieve discovered satellite products, optionally filtered by lake."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        if lake_id:
            rows = conn.execute(
                """
                SELECT * FROM satellite_products
                WHERE lake_id = ?
                ORDER BY acquisition_date DESC
                LIMIT ?
                """,
                (lake_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM satellite_products
                ORDER BY acquisition_date DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    results: List[SatelliteProduct] = []
    for r in rows:
        results.append(
            SatelliteProduct(
                product_id=r["product_id"],
                collection=r["collection"],
                acquisition_date=r["acquisition_date"],
                bbox=json.loads(r["bbox_json"]) if r["bbox_json"] else [],
                lake_id=r["lake_id"],
                cloud_cover_pct=r["cloud_cover_pct"],
                platform=r["platform"],
                orbit_direction=r["orbit_direction"],
                discovered_at=r["discovered_at"],
                metadata=json.loads(r["metadata_json"]) if r["metadata_json"] else {},
            )
        )
    return results


def product_exists(product_id: str, db_path: Optional[Path] = None) -> bool:
    """Check if a satellite product ID is already stored."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM satellite_products WHERE product_id = ?",
            (product_id,),
        ).fetchone()
        return row is not None


# ── Observations ─────────────────────────────────────────────────────────────

def save_observation(
    obs: ObservationRecord,
    db_path: Optional[Path] = None,
) -> None:
    """Insert or replace an observation record."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO observations (
                    observation_id, lake_id, sensor, acquisition_date,
                    product_id, area_0m_km2, area_50m_km2, valid_frac,
                    cloud_frac, vv_median_db, created_at, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    obs.observation_id,
                    obs.lake_id,
                    obs.sensor,
                    obs.acquisition_date,
                    obs.product_id,
                    obs.area_0m_km2,
                    obs.area_50m_km2,
                    obs.valid_frac,
                    obs.cloud_frac,
                    obs.vv_median_db,
                    obs.created_at or datetime.datetime.utcnow().isoformat() + "Z",
                    obs.notes,
                ),
            )


def get_observations(
    lake_id: Optional[str] = None,
    limit: int = 50,
    db_path: Optional[Path] = None,
) -> List[ObservationRecord]:
    """Retrieve observations for a lake or all lakes."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        if lake_id:
            rows = conn.execute(
                """
                SELECT * FROM observations
                WHERE lake_id = ?
                ORDER BY acquisition_date DESC
                LIMIT ?
                """,
                (lake_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM observations
                ORDER BY acquisition_date DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    results: List[ObservationRecord] = []
    for r in rows:
        results.append(
            ObservationRecord(
                observation_id=r["observation_id"],
                lake_id=r["lake_id"],
                sensor=r["sensor"],
                acquisition_date=r["acquisition_date"],
                product_id=r["product_id"],
                area_0m_km2=r["area_0m_km2"],
                area_50m_km2=r["area_50m_km2"],
                valid_frac=r["valid_frac"],
                cloud_frac=r["cloud_frac"],
                vv_median_db=r["vv_median_db"],
                created_at=r["created_at"],
                notes=r["notes"],
            )
        )
    return results


# ── Pipeline Runs ────────────────────────────────────────────────────────────

def save_pipeline_run(
    run: PipelineRun,
    db_path: Optional[Path] = None,
) -> None:
    """Insert or replace a pipeline run."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO pipeline_runs (
                    run_id, lake_id, start_time, end_time, stage,
                    status, input_products_json, model_features_json,
                    model_prediction, error_message, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.lake_id,
                    run.start_time,
                    run.end_time,
                    run.stage,
                    run.status,
                    json.dumps(run.input_product_ids),
                    json.dumps(run.model_features) if run.model_features else None,
                    run.model_prediction,
                    run.error_message,
                    json.dumps(run.metadata),
                ),
            )


def get_pipeline_runs(
    lake_id: Optional[str] = None,
    limit: int = 30,
    db_path: Optional[Path] = None,
) -> List[PipelineRun]:
    """Retrieve pipeline execution records."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        if lake_id:
            rows = conn.execute(
                """
                SELECT * FROM pipeline_runs
                WHERE lake_id = ?
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (lake_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM pipeline_runs
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    results: List[PipelineRun] = []
    for r in rows:
        results.append(
            PipelineRun(
                run_id=r["run_id"],
                lake_id=r["lake_id"],
                start_time=r["start_time"],
                end_time=r["end_time"],
                stage=r["stage"],
                status=r["status"],
                input_product_ids=json.loads(r["input_products_json"])
                if r["input_products_json"]
                else [],
                model_features=json.loads(r["model_features_json"])
                if r["model_features_json"]
                else None,
                model_prediction=r["model_prediction"],
                error_message=r["error_message"],
                metadata=json.loads(r["metadata_json"])
                if r["metadata_json"]
                else {},
            )
        )
    return results


# ── Monitoring Alerts & Status ───────────────────────────────────────────────

def save_monitoring_alert(
    alert: MonitoringAlert,
    db_path: Optional[Path] = None,
) -> None:
    """Insert or replace a monitoring status alert."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO monitoring_alerts (
                    alert_id, lake_id, timestamp, status, user_facing_status,
                    primary_area_km2, baseline_area_km2, area_change_km2,
                    area_change_pct, sensor_agreement_status,
                    cross_sensor_disagreement_pct, temporal_pair_days,
                    temporal_matching_flag, data_quality, staleness_days,
                    evidence_json, rationale, limitations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.alert_id,
                    alert.lake_id,
                    alert.timestamp,
                    alert.status,
                    alert.user_facing_status,
                    alert.primary_area_km2,
                    alert.baseline_area_km2,
                    alert.area_change_km2,
                    alert.area_change_pct,
                    alert.sensor_agreement_status,
                    alert.cross_sensor_disagreement_pct,
                    alert.temporal_pair_days,
                    alert.temporal_matching_flag,
                    alert.data_quality,
                    alert.staleness_days,
                    json.dumps(alert.evidence),
                    alert.rationale,
                    alert.limitations,
                ),
            )


def get_latest_monitoring_alerts(
    db_path: Optional[Path] = None,
) -> Dict[str, MonitoringAlert]:
    """Retrieve the most recent monitoring alert for each lake."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT m.*
            FROM monitoring_alerts m
            INNER JOIN (
                SELECT lake_id, MAX(timestamp) as max_time
                FROM monitoring_alerts
                GROUP BY lake_id
            ) latest ON m.lake_id = latest.lake_id AND m.timestamp = latest.max_time
            """
        ).fetchall()

    results: Dict[str, MonitoringAlert] = {}
    for r in rows:
        results[r["lake_id"]] = MonitoringAlert(
            alert_id=r["alert_id"],
            lake_id=r["lake_id"],
            timestamp=r["timestamp"],
            status=r["status"],
            user_facing_status=r["user_facing_status"],
            primary_area_km2=r["primary_area_km2"],
            baseline_area_km2=r["baseline_area_km2"],
            area_change_km2=r["area_change_km2"],
            area_change_pct=r["area_change_pct"],
            sensor_agreement_status=r["sensor_agreement_status"],
            cross_sensor_disagreement_pct=r["cross_sensor_disagreement_pct"],
            temporal_pair_days=r["temporal_pair_days"],
            temporal_matching_flag=r["temporal_matching_flag"],
            data_quality=r["data_quality"] or "LOW",
            staleness_days=r["staleness_days"],
            evidence=json.loads(r["evidence_json"]) if r["evidence_json"] else {},
            rationale=r["rationale"] or "",
            limitations=r["limitations"] or "",
        )
    return results


def get_monitoring_history(
    lake_id: str,
    limit: int = 50,
    db_path: Optional[Path] = None,
) -> List[MonitoringAlert]:
    """Retrieve status and alert history for a specific lake."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM monitoring_alerts
            WHERE lake_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (lake_id, limit),
        ).fetchall()

    history: List[MonitoringAlert] = []
    for r in rows:
        history.append(
            MonitoringAlert(
                alert_id=r["alert_id"],
                lake_id=r["lake_id"],
                timestamp=r["timestamp"],
                status=r["status"],
                user_facing_status=r["user_facing_status"],
                primary_area_km2=r["primary_area_km2"],
                baseline_area_km2=r["baseline_area_km2"],
                area_change_km2=r["area_change_km2"],
                area_change_pct=r["area_change_pct"],
                sensor_agreement_status=r["sensor_agreement_status"],
                cross_sensor_disagreement_pct=r["cross_sensor_disagreement_pct"],
                temporal_pair_days=r["temporal_pair_days"],
                temporal_matching_flag=r["temporal_matching_flag"],
                data_quality=r["data_quality"] or "LOW",
                staleness_days=r["staleness_days"],
                evidence=json.loads(r["evidence_json"]) if r["evidence_json"] else {},
                rationale=r["rationale"] or "",
                limitations=r["limitations"] or "",
            )
        )
    return history


def seed_from_qa_csv(
    csv_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Seed SQLite database with baseline data from NB10.2 QA CSV if empty."""
    init_db(db_path)
    with get_db_connection(db_path) as conn:
        existing = conn.execute("SELECT COUNT(*) FROM monitoring_alerts").fetchone()[0]
        if existing > 0:
            return 0  # Already seeded or has data

    from glacierguard.dashboard_data import load_qa_csv
    from glacierguard.risk_engine import map_status_to_user_facing

    try:
        df = load_qa_csv(csv_path)
    except Exception:
        return 0

    count = 0
    now_str = datetime.datetime.utcnow().isoformat() + "Z"
    for _, row in df.iterrows():
        lake_id = str(row["lake_id"])
        status = str(row["monitoring_status"])
        user_facing = map_status_to_user_facing(status)
        baseline = float(row["glo_2023_vector_ref_km2"])
        primary = float(row["primary_monitoring_area_km2"])
        diff_km2 = float(row["diff_from_glo_2023_km2"])
        diff_pct = float(row["diff_from_glo_2023_pct"])

        s2_date = str(row["latest_s2_date"]) if pd_not_na(row.get("latest_s2_date")) else None
        s1_date = str(row["latest_s1_date"]) if pd_not_na(row.get("latest_s1_date")) else None

        # Compute staleness
        from glacierguard.dashboard_data import compute_staleness_days
        staleness = compute_staleness_days(s2_date)

        # Create alert record
        alert = MonitoringAlert(
            alert_id=f"seed_{lake_id}_{s2_date or now_str[:10]}",
            lake_id=lake_id,
            timestamp=f"{s2_date or '2026-09-28'}T12:00:00Z",
            status=status,
            user_facing_status=user_facing,
            primary_area_km2=primary,
            baseline_area_km2=baseline,
            area_change_km2=diff_km2,
            area_change_pct=diff_pct,
            sensor_agreement_status=str(row.get("sensor_agreement_status", "")),
            cross_sensor_disagreement_pct=float(row["cross_sensor_disagreement_pct"])
            if pd_not_na(row.get("cross_sensor_disagreement_pct"))
            else None,
            temporal_pair_days=int(row["temporal_pair_days"])
            if pd_not_na(row.get("temporal_pair_days"))
            else None,
            temporal_matching_flag=str(row.get("temporal_matching_flag", "")),
            data_quality=str(row.get("data_quality", "MEDIUM")),
            staleness_days=staleness,
            evidence={
                "source": "10_2_five_lake_monitoring_qa.csv",
                "s2_date": s2_date,
                "s1_date": s1_date,
                "s2_area_0m_km2": float(row["s2_area_0m_km2"]) if pd_not_na(row.get("s2_area_0m_km2")) else None,
                "s1_area_0m_km2": float(row["s1_area_0m_km2"]) if pd_not_na(row.get("s1_area_0m_km2")) else None,
            },
            rationale=str(row.get("status_rationale", "")),
            limitations=str(row.get("extraction_limitations", "")),
        )
        save_monitoring_alert(alert, db_path)

        # Also create initial observation records
        if s2_date:
            obs_s2 = ObservationRecord(
                observation_id=f"seed_s2_{lake_id}_{s2_date}",
                lake_id=lake_id,
                sensor="SENTINEL-2",
                acquisition_date=s2_date,
                area_0m_km2=float(row["s2_area_0m_km2"]) if pd_not_na(row.get("s2_area_0m_km2")) else None,
                area_50m_km2=float(row["s2_area_50m_km2"]) if pd_not_na(row.get("s2_area_50m_km2")) else None,
                valid_frac=float(row["s2_valid_frac"]) if pd_not_na(row.get("s2_valid_frac")) else None,
                cloud_frac=float(row["s2_cloud_frac"]) if pd_not_na(row.get("s2_cloud_frac")) else None,
                notes="Seeded from research baseline QA CSV",
            )
            save_observation(obs_s2, db_path)

        if s1_date:
            obs_s1 = ObservationRecord(
                observation_id=f"seed_s1_{lake_id}_{s1_date}",
                lake_id=lake_id,
                sensor="SENTINEL-1",
                acquisition_date=s1_date,
                area_0m_km2=float(row["s1_area_0m_km2"]) if pd_not_na(row.get("s1_area_0m_km2")) else None,
                area_50m_km2=float(row["s1_area_50m_km2"]) if pd_not_na(row.get("s1_area_50m_km2")) else None,
                vv_median_db=float(row["s1_vv_median_db"]) if pd_not_na(row.get("s1_vv_median_db")) else None,
                notes="Seeded from research baseline QA CSV",
            )
            save_observation(obs_s1, db_path)

        count += 1

    return count


def pd_not_na(val: Any) -> bool:
    """Helper to check if a value is not NA/null."""
    if val is None:
        return False
    try:
        import numpy as np
        if np.isnan(val):
            return False
    except (TypeError, ValueError):
        pass
    s = str(val).strip().lower()
    return s not in ("", "nan", "none", "null")
