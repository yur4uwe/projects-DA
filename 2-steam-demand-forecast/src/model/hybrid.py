"""Hybrid Prophet + LightGBM model conforming directly to course Listing 5."""

import logging

import numpy as np
import pandas as pd
from prophet import Prophet
from src.models.lgbm_model import LightGBMDemandModel

logger = logging.getLogger(__name__)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class HybridProphetLGBMModel:
    """M4 Hybrid Model: Prophet for baseline & seasonality + LightGBM for residual correction."""

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        lgb_n_estimators: int = 400,
        lgb_learning_rate: float = 0.03,
        max_history_days: int = 730,
        random_state: int = 42,
    ) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.max_history_days = max_history_days
        self.random_state = random_state

        self.prophet_model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        # Add required external regressors as specified in Listing 5
        self.prophet_model.add_regressor("promotion")
        self.prophet_model.add_regressor("holiday")

        # LightGBM to model residuals (no log transform on residuals, and allow negative predictions)
        self.lgb_residual_model = LightGBMDemandModel(
            n_estimators=lgb_n_estimators,
            learning_rate=lgb_learning_rate,
            log_transform=False,
            clip_non_negative=False,
            random_state=random_state,
        )

    def _prepare_prophet_df(
        self, df: pd.DataFrame, target_col: str | None = None
    ) -> pd.DataFrame:
        """Format input dataframe to Prophet schema ('ds', 'y', 'promotion', 'holiday')."""
        pdf = pd.DataFrame()
        pdf["ds"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        if target_col and target_col in df.columns:
            # Fit in log space to prevent trend explosion on high demand peaks
            pdf["y"] = np.log1p(df[target_col].astype(float))

        pdf["promotion"] = (
            df["discount_depth"].astype(float)
            if "discount_depth" in df.columns
            else 0.0
        )
        pdf["holiday"] = (
            df["is_steam_sale"].astype(float) if "is_steam_sale" in df.columns else 0.0
        )
        return pdf

    def fit(
        self, df: pd.DataFrame, target_col: str = "demand"
    ) -> "HybridProphetLGBMModel":
        """Fit Prophet baseline and train LightGBM on Prophet prediction residuals."""
        # Trim training data to recent window (e.g. 2 years) for stability
        fit_df = (
            df.iloc[-self.max_history_days :].copy()
            if len(df) > self.max_history_days
            else df.copy()
        )

        prophet_data = self._prepare_prophet_df(fit_df, target_col=target_col)
        self.prophet_model.fit(prophet_data)

        # 1. In-sample Prophet baseline predictions (transformed back to linear scale)
        prophet_forecast = self.prophet_model.predict(prophet_data)
        prophet_base = np.expm1(prophet_forecast["yhat"].values)

        # 2. Compute residuals: residual = y_true - yhat_prophet
        y_true = fit_df[target_col].values
        residuals = y_true - prophet_base

        # 3. Fit LightGBM on residuals using lag and calendar features
        self.lgb_residual_model.fit(fit_df, residuals)
        return self

    def predict_components(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Generate decomposed components: Prophet baseline, LightGBM residual, and total."""
        prophet_data = self._prepare_prophet_df(df)
        prophet_forecast = self.prophet_model.predict(prophet_data)

        prophet_base = np.clip(np.expm1(prophet_forecast["yhat"].values), 0.0, None)
        lgb_residual = self.lgb_residual_model.predict(df)

        combined = np.clip(prophet_base + lgb_residual, 0.0, None)
        yhat_lower = np.clip(
            np.expm1(prophet_forecast["yhat_lower"].values) + lgb_residual * 0.8,
            0.0,
            None,
        )
        yhat_upper = np.clip(
            np.expm1(prophet_forecast["yhat_upper"].values) + lgb_residual * 1.2,
            0.0,
            None,
        )

        return {
            "combined": combined,
            "prophet_base": prophet_base,
            "lgbm_residual": lgb_residual,
            "trend": prophet_forecast.get("trend", prophet_base).values,
            "weekly": prophet_forecast.get("weekly", np.zeros(len(df))).values,
            "yearly": prophet_forecast.get("yearly", np.zeros(len(df))).values,
            "yhat_lower": yhat_lower,
            "yhat_upper": yhat_upper,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return combined final calibrated demand predictions."""
        return self.predict_components(df)["combined"]
