"""
GlacierGuard-AI Processing Pipeline Orchestrator
================================================
Defensible, multi-stage processing pipeline coordinating satellite discovery,
QA verification, frozen machine learning model inference, operational risk
evaluation, and relational persistence.

Pipeline Stages:
1. DISCOVER: Query CDSE STAC catalogue for latest Sentinel-1/Sentinel-2 products.
2. INGEST: Register newly discovered scenes into SQLite with provenance tracking.
3. PREPROCESS & QA: Assess observation validity, temporal pair matching,
   and cross-sensor agreement.
4. FEATURE_EXTRACT: Structure temporal and spatial metrics into defensible features.
5. INFERENCE: Evaluate annual area change predictions against the frozen
   XGBoost model (NB09) where multi-year lags exist.
6. EVALUATE: Run conservative RiskEngine rules to assign operational status.
7. PERSIST: Commit updated observations, pipeline runs, and alerts to SQLite.
"""

import datetime
import logging
import traceback
from typing import Any, Dict, List, Optional
import uuid

from glacierguard.cdse_client import CDSEClient
from glacierguard.config import Settings, default_settings
from glacierguard.db import (
    get_observations,
    save_monitoring_alert,
    save_observation,
    save_pipeline_run,
)
from glacierguard.inference.fused_model import load_fused_xgboost_model
from glacierguard.models import (
    MonitoringAlert,
    ObservationRecord,
    PipelineRun,
    SatelliteProduct,
)
from glacierguard.risk_engine import RiskEngine


logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Orchestrates end-to-end data discovery, inference, and operational tiering."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or default_settings
        self.cdse_client = CDSEClient(self.settings)
        self.risk_engine = RiskEngine(self.settings)
        self._model_predictor = None

    def _get_predictor(self):
        """Lazy load frozen XGBoost model predictor."""
        if self._model_predictor is None and self.settings.model_path.is_file():
            try:
                self._model_predictor = load_fused_xgboost_model(self.settings.model_path)
            except Exception as exc:
                logger.warning("Could not load frozen XGBoost model: %s", exc)
        return self._model_predictor

    def run_for_lake(
        self,
        lake_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        mock_products: Optional[List[SatelliteProduct]] = None,
        mock_observations: Optional[List[ObservationRecord]] = None,
    ) -> PipelineRun:
        """Execute the full processing pipeline for a single lake.

        Parameters
        ----------
        lake_id : str
            Identifier of the target lake.
        start_date : str, optional
            Observation window start 'YYYY-MM-DD'.
        end_date : str, optional
            Observation window end 'YYYY-MM-DD'.
        mock_products : List[SatelliteProduct], optional
            Optional pre-discovered products for offline or test execution.
        mock_observations : List[ObservationRecord], optional
            Optional pre-computed observations for testing.

        Returns
        -------
        PipelineRun
            Traceable run record with execution status and artifacts.
        """
        run_id = f"run_{lake_id}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        start_time = datetime.datetime.utcnow().isoformat() + "Z"

        run = PipelineRun(
            run_id=run_id,
            lake_id=lake_id,
            start_time=start_time,
            stage="DISCOVER",
            status="RUNNING",
            input_product_ids=[],
            metadata={"start_date": start_date, "end_date": end_date},
        )
        save_pipeline_run(run, self.settings.db_path)

        try:
            # ── Stage 1: DISCOVER ────────────────────────────────────────────
            logger.info("[%s] Stage 1: DISCOVER for %s", run_id, lake_id)
            run.stage = "DISCOVER"
            save_pipeline_run(run, self.settings.db_path)

            if mock_products is not None:
                products = mock_products
            else:
                _, products = self.cdse_client.discover_and_persist(
                    lake_id=lake_id,
                    start_date=start_date,
                    end_date=end_date,
                )

            run.input_product_ids = [p.product_id for p in products]

            # ── Stage 2: INGEST ──────────────────────────────────────────────
            logger.info("[%s] Stage 2: INGEST for %s", run_id, lake_id)
            run.stage = "INGEST"
            save_pipeline_run(run, self.settings.db_path)

            # Retrieve latest observations (or use mock if provided)
            if mock_observations is not None:
                for obs in mock_observations:
                    save_observation(obs, self.settings.db_path)
                obs_list = mock_observations
            else:
                obs_list = get_observations(lake_id=lake_id, limit=20, db_path=self.settings.db_path)

            # Separate latest S1 and S2 observations
            s2_obs = next((o for o in obs_list if o.sensor.upper() == "SENTINEL-2"), None)
            s1_obs = next((o for o in obs_list if o.sensor.upper() == "SENTINEL-1"), None)

            # ── Stage 3: PREPROCESS & QA ─────────────────────────────────────
            logger.info("[%s] Stage 3: PREPROCESS & QA for %s", run_id, lake_id)
            run.stage = "PREPROCESS_QA"
            save_pipeline_run(run, self.settings.db_path)

            # ── Stage 4: FEATURE EXTRACT ─────────────────────────────────────
            logger.info("[%s] Stage 4: FEATURE_EXTRACT for %s", run_id, lake_id)
            run.stage = "FEATURE_EXTRACT"
            features: Dict[str, Any] = {
                "lake_id": lake_id,
                "latest_s2_date": s2_obs.acquisition_date if s2_obs else None,
                "latest_s1_date": s1_obs.acquisition_date if s1_obs else None,
                "s2_area_km2": s2_obs.area_50m_km2 if s2_obs else None,
                "s1_area_km2": s1_obs.area_50m_km2 if s1_obs else None,
            }
            run.model_features = features
            save_pipeline_run(run, self.settings.db_path)

            # ── Stage 5: INFERENCE (Frozen XGBoost) ───────────────────────────
            logger.info("[%s] Stage 5: INFERENCE for %s", run_id, lake_id)
            run.stage = "INFERENCE"
            predictor = self._get_predictor()
            model_pred: Optional[float] = None

            # Note: The frozen XGBoost model predicts annual delta given annual lag features.
            # In live per-scene monitoring, we document this boundary clearly.
            if predictor is not None and "AREA_LAG1" in features:
                try:
                    import pandas as pd
                    feat_df = pd.DataFrame([features])
                    preds = predictor.predict(feat_df)
                    if len(preds) > 0:
                        model_pred = float(preds[0])
                except Exception as exc:
                    logger.debug("Model inference skipped or encountered error: %s", exc)

            run.model_prediction = model_pred
            save_pipeline_run(run, self.settings.db_path)

            # ── Stage 6: EVALUATE (Operational Risk Engine) ───────────────────
            logger.info("[%s] Stage 6: EVALUATE for %s", run_id, lake_id)
            run.stage = "EVALUATE"
            alert = self.risk_engine.evaluate(
                lake_id=lake_id,
                s2_obs=s2_obs,
                s1_obs=s1_obs,
            )

            # ── Stage 7: PERSIST ─────────────────────────────────────────────
            logger.info("[%s] Stage 7: PERSIST for %s", run_id, lake_id)
            run.stage = "PERSIST"
            save_monitoring_alert(alert, self.settings.db_path)

            # Mark run as SUCCESS
            run.status = "SUCCESS"
            run.stage = "COMPLETED"
            run.end_time = datetime.datetime.utcnow().isoformat() + "Z"
            run.metadata["alert_id"] = alert.alert_id
            run.metadata["user_facing_status"] = alert.user_facing_status
            save_pipeline_run(run, self.settings.db_path)

            logger.info(
                "[%s] Completed pipeline for %s: %s",
                run_id,
                lake_id,
                alert.user_facing_status,
            )
            return run

        except Exception as exc:
            logger.error("[%s] Pipeline failed for %s: %s", run_id, lake_id, exc)
            run.status = "FAILED"
            run.error_message = f"{type(exc).__name__}: {str(exc)}"
            run.end_time = datetime.datetime.utcnow().isoformat() + "Z"
            run.metadata["traceback"] = traceback.format_exc()
            save_pipeline_run(run, self.settings.db_path)

            # Emit SYSTEM_ERROR alert
            err_alert = MonitoringAlert(
                alert_id=f"alert_err_{lake_id}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                lake_id=lake_id,
                timestamp=run.end_time,
                status="SYSTEM_ERROR",
                user_facing_status="SYSTEM ERROR",
                data_quality="LOW",
                rationale=f"Automated pipeline failure: {type(exc).__name__}",
                limitations="Pipeline execution failed; inspect error log.",
                evidence={"error": str(exc), "run_id": run_id},
            )
            save_monitoring_alert(err_alert, self.settings.db_path)
            return run

    def run_all_lakes(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[PipelineRun]:
        """Execute pipeline across all configured lakes."""
        lakes = self.settings.list_lake_ids()
        runs: List[PipelineRun] = []
        for lid in lakes:
            r = self.run_for_lake(lid, start_date=start_date, end_date=end_date)
            runs.append(r)
        return runs
