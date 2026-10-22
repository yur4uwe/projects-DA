"""Decoupled Two-Stage Hybrid Demand & Promotional Uplift Model conforming to Listing 5."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import Ridge

from src.features.constants import (
    DEFAULT_ELASTICITY_EPSILON,
    SALE_RAMP,
    WEEKEND_BOOST,
)
from src.model.lgbm_model import LightGBMDemandModel

logger = logging.getLogger(__name__)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class HybridProphetLGBMModel:
    """M4 Hybrid Architecture with Decoupled Organic Baseline and Scale-Invariant Uplift.

    Stage 1 (Organic):
      Prophet macro baseline on normalized demand (rolling-30 scale) +
      Scale-invariant LightGBM model on ratio-normalized lags and rolling statistics.
      Target: log(y / prophet_base) on unpromoted observations.

    Stage 2 (Uplift):
      Uplift regressor on structural promotional features and theoretical price elasticity.
      Target: (y / organic_demand) - 1.0 on promotional observations.
      Guarantees scale-invariance and monotonic response across discount depth.
    """

    ORGANIC_FEATURE_COLS = [
        "norm_lag_1",
        "norm_lag_2",
        "norm_lag_7",
        "norm_lag_14",
        "norm_lag_30",
        "norm_rolling_3",
        "norm_rolling_std_3",
        "norm_rolling_7",
        "norm_rolling_std_7",
        "norm_rolling_14",
        "norm_rolling_std_14",
        "norm_rolling_std_30",
        "sales_momentum_3_7",
        "sales_momentum_7_30",
        "sales_trend_3d",
        "day_of_week",
        "is_weekend",
        "month",
        "sin_day_of_week",
        "cos_day_of_week",
        "sin_day_of_year",
        "cos_day_of_year",
        "base_price",
        "is_multiplayer",
        "is_coop",
        "log_positive_reviews",
    ]

    UPLIFT_FEATURE_COLS = [
        "base_elasticity",
        "discount_depth",
        "storefront_exposure",
        "is_weekend",
        "is_major_seasonal_sale",
        "sale_progress",
        "days_until_sale_end",
        "base_price",
        "is_multiplayer",
        "is_coop",
        "log_positive_reviews",
    ]

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        lgb_n_estimators: int = 400,
        lgb_learning_rate: float = 0.03,
        random_state: int = 42,
    ) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.random_state = random_state

        # Macro Prophet model (pure organic seasonality without promotional regressor pollution)
        self.prophet_model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale,
        )

        # Scale-invariant LightGBM for organic correction
        self.lgb_organic_model = LightGBMDemandModel(
            n_estimators=lgb_n_estimators,
            learning_rate=lgb_learning_rate,
            num_leaves=31,
            log_transform=False,
            clip_non_negative=False,
            feature_cols=self.ORGANIC_FEATURE_COLS,
            random_state=random_state,
        )

        # Uplift model: structural price elasticity and promotional exposure with strict non-negative monotonicity
        self.uplift_model = Ridge(alpha=10.0, positive=True, random_state=random_state)

    def _prepare_organic_features(
        self, df: pd.DataFrame, scale: pd.Series | np.ndarray
    ) -> pd.DataFrame:
        """Create scale-invariant normalized lag and rolling features."""
        X = pd.DataFrame(index=df.index)
        scale_safe = np.maximum(scale, 10.0)

        for k in [1, 2, 7, 14, 30]:
            col = f"sales_lag_{k}"
            X[f"norm_lag_{k}"] = df[col].values / scale_safe if col in df.columns else 1.0

        for w in [3, 7, 14]:
            col = f"sales_rolling_{w}"
            X[f"norm_rolling_{w}"] = df[col].values / scale_safe if col in df.columns else 1.0
            col_std = f"sales_rolling_std_{w}"
            X[f"norm_rolling_std_{w}"] = df[col_std].values / scale_safe if col_std in df.columns else 0.0

        if "sales_rolling_std_30" in df.columns:
            X["norm_rolling_std_30"] = df["sales_rolling_std_30"].values / scale_safe
        else:
            X["norm_rolling_std_30"] = 0.0

        for col in [
            "sales_momentum_3_7",
            "sales_momentum_7_30",
            "sales_trend_3d",
            "day_of_week",
            "is_weekend",
            "month",
            "sin_day_of_week",
            "cos_day_of_week",
            "sin_day_of_year",
            "cos_day_of_year",
            "base_price",
            "is_multiplayer",
            "is_coop",
            "log_positive_reviews",
        ]:
            X[col] = df[col].values if col in df.columns else 0.0

        return X

    def _prepare_uplift_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create promo-structural features with continuous economic elasticity."""
        X = pd.DataFrame(index=df.index)
        disc = df["discount_depth"].clip(0.0, 0.95) if "discount_depth" in df.columns else pd.Series(0.0, index=df.index)
        exposure = df["storefront_exposure"] if "storefront_exposure" in df.columns else pd.Series(1.0, index=df.index)
        weekend = df["is_weekend"] if "is_weekend" in df.columns else pd.Series(0, index=df.index)

        # Theoretical economic price elasticity base: ((1 - d)^(-epsilon) - 1.0) * exposure * weekend_boost
        elast_lift = (1.0 - disc).pow(-DEFAULT_ELASTICITY_EPSILON) - 1.0
        weekend_factor = 1.0 + WEEKEND_BOOST * weekend
        X["base_elasticity"] = elast_lift * exposure * weekend_factor
        X["discount_depth"] = disc
        X["storefront_exposure"] = exposure
        X["is_weekend"] = weekend
        X["is_major_seasonal_sale"] = df["is_major_seasonal_sale"].values if "is_major_seasonal_sale" in df.columns else 0
        X["sale_progress"] = df["sale_progress"].values if "sale_progress" in df.columns else 0.0
        X["days_until_sale_end"] = df["days_until_sale_end"].values if "days_until_sale_end" in df.columns else 0.0

        for col in ["base_price", "is_multiplayer", "is_coop", "log_positive_reviews"]:
            X[col] = df[col].values if col in df.columns else 0.0

        return X

    def fit(
        self, df: pd.DataFrame, target_col: str = "demand"
    ) -> "HybridProphetLGBMModel":
        """Fit two-stage decoupled model on historical data."""
        fit_df = df.copy()
        fit_df["date"] = pd.to_datetime(fit_df["date"])

        # Determine dynamic baseline scale per observation
        if "sales_rolling_30" in fit_df.columns and fit_df["sales_rolling_30"].notna().any():
            scale = fit_df["sales_rolling_30"].clip(lower=10.0)
        elif "sales_lag_1" in fit_df.columns and fit_df["sales_lag_1"].notna().any():
            scale = fit_df["sales_lag_1"].clip(lower=10.0)
        else:
            scale = pd.Series(fit_df[target_col].median(), index=fit_df.index).clip(lower=10.0)

        fit_df["_scale"] = scale
        fit_df["_norm_demand"] = fit_df[target_col] / fit_df["_scale"]

        # 1. Fit Prophet macro model strictly on unpromoted / organic series
        is_sale = fit_df["is_steam_sale"] == 1 if "is_steam_sale" in fit_df.columns else pd.Series(False, index=fit_df.index)
        non_sale_df = fit_df[~is_sale]
        macro_df = (
            non_sale_df.groupby("date")["_norm_demand"]
            .mean()
            .reset_index()
        )

        macro_prophet_data = pd.DataFrame()
        macro_prophet_data["ds"] = macro_df["date"].dt.tz_localize(None)
        macro_prophet_data["y"] = np.log(macro_df["_norm_demand"].clip(lower=0.01).astype(float))
        self.prophet_model.fit(macro_prophet_data)

        # In-sample Prophet baseline predictions across full training set
        prophet_in = pd.DataFrame()
        prophet_in["ds"] = fit_df["date"].dt.tz_localize(None)
        prophet_forecast = self.prophet_model.predict(prophet_in)
        norm_base = np.clip(np.exp(prophet_forecast["yhat"].values), 0.0, None)
        prophet_base = norm_base * fit_df["_scale"].values

        # 2. Fit Stage 1 Organic LightGBM model on organic (non-sale) rows
        X_org = self._prepare_organic_features(fit_df, scale.values)
        log_org_ratio = np.log(np.clip(fit_df[target_col].values / np.maximum(prophet_base, 10.0), 0.05, 20.0))
        self.lgb_organic_model.fit(X_org[~is_sale], log_org_ratio[~is_sale])

        # Predict full in-sample organic demand
        pred_org_full = prophet_base * np.exp(self.lgb_organic_model.predict(X_org))

        # 3. Fit Stage 2 Uplift model on promotional rows
        if is_sale.any():
            X_up = self._prepare_uplift_features(fit_df)
            y_actual = fit_df[target_col].values
            rel_uplift_target = np.clip((y_actual[is_sale] / np.maximum(prophet_base[is_sale], 10.0)) - 1.0, 0.0, 10.0)
            self.uplift_model.fit(X_up[is_sale], rel_uplift_target)

        return self

    def predict_components(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Generate decomposed components: organic baseline, uplift factor, and combined."""
        if "sales_rolling_30" in df.columns and df["sales_rolling_30"].notna().any():
            scale = df["sales_rolling_30"].clip(lower=10.0).values
        elif "sales_lag_1" in df.columns and df["sales_lag_1"].notna().any():
            scale = df["sales_lag_1"].clip(lower=10.0).values
        else:
            scale = np.ones(len(df))

        prophet_data = pd.DataFrame()
        prophet_data["ds"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        prophet_forecast = self.prophet_model.predict(prophet_data)

        norm_base = np.clip(np.exp(prophet_forecast["yhat"].values), 0.0, None)
        prophet_base = norm_base * scale

        # Organic correction
        X_org = self._prepare_organic_features(df, scale)
        r_org = self.lgb_organic_model.predict(X_org)
        organic_demand = np.clip(prophet_base * np.exp(r_org), 0.0, None)

        # Uplift calculation
        is_sale = df["is_steam_sale"] == 1 if "is_steam_sale" in df.columns else np.zeros(len(df), dtype=bool)
        if isinstance(is_sale, pd.Series):
            is_sale = is_sale.values

        if np.any(is_sale):
            X_up = self._prepare_uplift_features(df)
            rel_lift = np.maximum(0.0, self.uplift_model.predict(X_up))
            uplift_mult = np.where(is_sale, 1.0 + rel_lift, 1.0)
            combined = organic_demand * uplift_mult
        else:
            combined = organic_demand
            uplift_mult = np.ones(len(df))

        return {
            "combined": combined,
            "prophet_base": prophet_base,
            "organic_demand": organic_demand,
            "uplift_factor": uplift_mult,
            "lgbm_residual": combined - prophet_base,
            "trend": prophet_forecast.get("trend", prophet_base).values,
            "weekly": prophet_forecast.get("weekly", np.zeros(len(df))).values,
            "yearly": prophet_forecast.get("yearly", np.zeros(len(df))).values,
            "yhat_lower": np.clip(combined * 0.85, 0.0, None),
            "yhat_upper": combined * 1.15,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return combined demand predictions."""
        return self.predict_components(df)["combined"]
