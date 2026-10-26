"""
Inference Interface for Frozen Fused-XGBoost Model
===================================================
Provides reproducible prediction and evaluation using the frozen NB09
dual-sensor fused XGBoost regression artifact.
"""

from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_MODEL_PATH = Path("reports/models/09_final_fused_xgboost.pkl")


class FusedXGBoostPredictor:
    """Wrapper for the frozen NB09 dual-sensor fused XGBoost regression model.

    Preserves the benchmark contract (test MAE = 0.00479 km²) and encapsulates
    the train-only fitted ColumnTransformer and XGBRegressor.
    """

    def __init__(self, model_dict_or_path: Union[str, Path, Dict[str, Any]]):
        if isinstance(model_dict_or_path, (str, Path)):
            path = Path(model_dict_or_path)
            if not path.is_file():
                raise FileNotFoundError(f"Model artifact not found at: {path}")
            with open(path, 'rb') as f:
                data = pickle.load(f)
        elif isinstance(model_dict_or_path, dict):
            data = model_dict_or_path
        else:
            raise TypeError(
                f"Expected str, Path, or dict, got {type(model_dict_or_path)}"
            )

        self._raw_dict = data
        self.model = data.get('model')
        self.preprocessor = data.get('preprocessor')
        self.feature_num_cols: List[str] = data.get('feature_num_cols', [])
        self.feature_cat_cols: List[str] = data.get('feature_cat_cols', [])
        self.required_features: List[str] = self.feature_cat_cols + self.feature_num_cols
        self.frozen_test_mae: float = float(data.get('test_mae', np.nan))
        self.frozen_test_rmse: float = float(data.get('test_rmse', np.nan))
        self.hyperparameters: Dict[str, Any] = data.get('hyperparameters', {})
        self.training_period: str = data.get('training_period', '2017-2021')
        self.validation_period: str = data.get('validation_period', '2022')
        self.test_period: str = data.get('test_period', '2023')

        if self.model is None or self.preprocessor is None:
            raise ValueError("Model dictionary must contain both 'model' and 'preprocessor'.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input features and generate lake area-change predictions [km²].

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing all required categorical and numerical features.

        Returns
        -------
        np.ndarray
            Predicted next-year area change values in km².
        """
        missing = [col for col in self.required_features if col not in X.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing}")

        X_subset = X[self.required_features]
        X_trans = self.preprocessor.transform(X_subset)
        preds = self.model.predict(X_trans)
        return np.asarray(preds, dtype=float)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> Dict[str, float]:
        """Compute MAE, RMSE, and R² metrics on a given evaluation set.

        Parameters
        ----------
        X : pd.DataFrame
            Feature observations.
        y : Union[pd.Series, np.ndarray]
            Ground truth NEXT_AREA_CHANGE values.

        Returns
        -------
        Dict[str, float]
            Dictionary containing 'MAE', 'RMSE', and 'R2'.
        """
        preds = self.predict(X)
        y_true = np.asarray(y, dtype=float)

        mae = float(mean_absolute_error(y_true, preds))
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        r2 = float(r2_score(y_true, preds))

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
        }


def load_fused_xgboost_model(
    model_path: Optional[Union[str, Path]] = None,
) -> FusedXGBoostPredictor:
    """Convenience factory to load the frozen fused XGBoost model."""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    return FusedXGBoostPredictor(model_path)
