"""Hybrid Prophet + LightGBM model conforming directly to course Listing 5."""

import logging

import numpy as np
import pandas as pd
from prophet import Prophet

from src.model.lgbm_model import LightGBMDemandModel

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
        # Add required external regressors as specified in Listing 5 (standardize=False ensures 0.0 is baseline)
        self.prophet_model.add_regressor("promotion", standardize=False)
        self.prophet_model.add_regressor("holiday", standardize=False)

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
        """Fit Prophet baseline on macro normalized demand and train LightGBM on Prophet prediction residuals."""
        fit_df = df.copy()
        fit_df["date"] = pd.to_datetime(fit_df["date"])

        # Determine dynamic baseline scale per observation
        if "sales_rolling_30" in fit_df.columns and fit_df["sales_rolling_30"].notna().any():
            scale = fit_df["sales_rolling_30"].clip(lower=10.0)
        elif "sales_lag_1" in fit_df.columns and fit_df["sales_lag_1"].notna().any():
            scale = fit_df["sales_lag_1"].clip(lower=10.0)
        elif "appid" in fit_df.columns:
            scale = fit_df.groupby("appid")[target_col].transform("median").clip(lower=10.0)
        else:
            scale = pd.Series(fit_df[target_col].median(), index=fit_df.index).clip(lower=10.0)

        fit_df["_scale"] = scale
        fit_df["_norm_demand"] = fit_df[target_col] / fit_df["_scale"]

        # Macro series across all active catalog games for Prophet over the historical span
        macro_df = (
            fit_df.groupby("date")
            .agg(
                {
                    "_norm_demand": "mean",
                    "discount_depth": "mean" if "discount_depth" in fit_df.columns else lambda x: 0.0,
                    "is_steam_sale": "max" if "is_steam_sale" in fit_df.columns else lambda x: 0.0,
                }
            )
            .reset_index()
        )

        macro_prophet_data = pd.DataFrame()
        macro_prophet_data["ds"] = macro_df["date"].dt.tz_localize(None)
        macro_prophet_data["y"] = np.log(macro_df["_norm_demand"].clip(lower=0.01).astype(float))
        is_sale_macro = macro_df["is_steam_sale"].astype(float) if "is_steam_sale" in macro_df.columns else 0.0
        disc_macro = macro_df["discount_depth"].astype(float) if "discount_depth" in macro_df.columns else 0.0
        macro_prophet_data["promotion"] = is_sale_macro * disc_macro
        macro_prophet_data["holiday"] = is_sale_macro

        self.prophet_model.fit(macro_prophet_data)

        # 1. In-sample Prophet baseline predictions across full training set
        is_sale_fit = fit_df["is_steam_sale"].astype(float) if "is_steam_sale" in fit_df.columns else 0.0
        disc_fit = fit_df["discount_depth"].astype(float) if "discount_depth" in fit_df.columns else 0.0
        prophet_in = pd.DataFrame()
        prophet_in["ds"] = fit_df["date"].dt.tz_localize(None)
        prophet_in["promotion"] = is_sale_fit * disc_fit
        prophet_in["holiday"] = is_sale_fit
        prophet_forecast = self.prophet_model.predict(prophet_in)

        norm_base = np.clip(np.exp(prophet_forecast["yhat"].values), 0.0, None)
        prophet_base = norm_base * fit_df["_scale"].values

        # 2. Compute residuals on native scale: residual = y_true - yhat_prophet
        y_true = fit_df[target_col].values
        residuals = y_true - prophet_base

        # 3. Fit LightGBM on residuals across all games
        self.lgb_residual_model.fit(fit_df, residuals)
        return self

    def predict_components(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Generate decomposed components: Prophet baseline, LightGBM residual, and total."""
        # Determine scale
        if "sales_rolling_30" in df.columns and df["sales_rolling_30"].notna().any():
            scale = df["sales_rolling_30"].clip(lower=10.0).values
        elif "sales_lag_1" in df.columns and df["sales_lag_1"].notna().any():
            scale = df["sales_lag_1"].clip(lower=10.0).values
        else:
            scale = np.ones(len(df))

        is_sale_p = df["is_steam_sale"].astype(float) if "is_steam_sale" in df.columns else 0.0
        disc_p = df["discount_depth"].astype(float) if "discount_depth" in df.columns else 0.0
        prophet_data = pd.DataFrame()
        prophet_data["ds"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        prophet_data["promotion"] = is_sale_p * disc_p
        prophet_data["holiday"] = is_sale_p
        prophet_forecast = self.prophet_model.predict(prophet_data)

        norm_base = np.clip(np.exp(prophet_forecast["yhat"].values), 0.0, None)
        prophet_base = norm_base * scale

        lgb_residual = self.lgb_residual_model.predict(df)

        combined = np.clip(prophet_base + lgb_residual, 0.0, None)
        norm_lower = np.clip(np.exp(prophet_forecast["yhat_lower"].values), 0.0, None) * scale
        norm_upper = np.clip(np.exp(prophet_forecast["yhat_upper"].values), 0.0, None) * scale
        yhat_lower = np.clip(norm_lower + lgb_residual * 0.8, 0.0, None)
        yhat_upper = np.clip(norm_upper + lgb_residual * 1.2, 0.0, None)

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
