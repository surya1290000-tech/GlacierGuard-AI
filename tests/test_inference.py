"""
Unit and Parity Tests for Frozen Fused-XGBoost Inference
========================================================
Verifies parity with NB09 dual-sensor fused XGBoost regression benchmark.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from glacierguard.inference.fused_model import (
    DEFAULT_MODEL_PATH,
    FusedXGBoostPredictor,
    load_fused_xgboost_model,
)

ROOT = Path(__file__).resolve().parent.parent


class TestFusedXGBoostInference:
    """Validate loading, metadata integrity, and inference execution."""

    @pytest.fixture(scope='class')
    def predictor(self):
        """Load frozen model once for all tests in class."""
        model_path = ROOT / DEFAULT_MODEL_PATH
        if not model_path.exists():
            pytest.skip(f"Model artifact not found at {model_path}")
        return load_fused_xgboost_model(model_path)

    def test_model_metadata(self, predictor):
        assert len(predictor.feature_cat_cols) == 4
        assert len(predictor.feature_num_cols) == 32
        assert len(predictor.required_features) == 36
        assert predictor.training_period == '2017-2021'
        assert predictor.validation_period == '2022'
        assert predictor.test_period == '2023'
        assert predictor.frozen_test_mae == pytest.approx(0.004792, abs=1e-5)
        assert predictor.frozen_test_rmse == pytest.approx(0.015079, abs=1e-5)

    def test_missing_features_raise_error(self, predictor):
        incomplete_df = pd.DataFrame({
            'AREA': [1.0, 2.0],
            'BASIN': ['A', 'B'],
        })
        with pytest.raises(ValueError, match="missing required features"):
            predictor.predict(incomplete_df)

    def test_predict_on_synthetic_data(self, predictor):
        # Create a single synthetic row with all 36 required features
        row = {}
        for cat in predictor.feature_cat_cols:
            row[cat] = ['Nepal' if cat == 'COUNTRY' else 'KOSHI']
        for num in predictor.feature_num_cols:
            row[num] = [1.0 if 'AREA' in num else 5000.0 if 'ELEV' in num else 0.0]

        df = pd.DataFrame(row)
        preds = predictor.predict(df)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == 1
        assert np.isfinite(preds[0])

    def test_evaluate_method(self, predictor):
        # Create small test dataset
        rows = []
        for i in range(5):
            r = {}
            for cat in predictor.feature_cat_cols:
                r[cat] = 'Nepal' if cat == 'COUNTRY' else 'KOSHI'
            for num in predictor.feature_num_cols:
                r[num] = 0.5 + i * 0.1
            rows.append(r)
        df = pd.DataFrame(rows)
        y = np.array([0.01, 0.02, 0.015, -0.01, 0.005])

        metrics = predictor.evaluate(df, y)
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        assert 'R2' in metrics
        assert metrics['MAE'] >= 0.0
        assert metrics['RMSE'] >= 0.0
