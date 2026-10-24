"""Decoupled Two-Stage Hybrid Demand & Promotional Uplift Model with Game-Specific Calibration."""

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


class SingleGameHybridModel:
    """Game-specific hybrid demand model fitting Prophet weekly cycles and LightGBM micro-corrections."""

    def __init__(
        self,
        appid: int = 0,
        game_name: str = "",
        changepoint_prior_scale: float = 0.01,
        weekly_seasonality: bool = True,
        lgb_n_estimators: int = 150,
        lgb_learning_rate: float = 0.03,
        random_state: int = 42,
        fallback_uplift_model: Any = None,
    ) -> None:
        self.appid = appid
        self.game_name = game_name
        self.weekly_seasonality = weekly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.lgb_n_estimators = lgb_n_estimators
        self.lgb_learning_rate = lgb_learning_rate
        self.random_state = random_state
        self.fallback_uplift_model = fallback_uplift_model
        self.peak_ratio: float = 1.45
        self.base_rolling_30: float = 5000.0
        self.uplift_model = Ridge(alpha=10.0, positive=True, random_state=random_state)
        self.has_empirical_sales: bool = False
        self.empirical_mean_lift: float = 0.0

        self.prophet_model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        self.lgb_organic_model = LightGBMDemandModel(
            n_estimators=lgb_n_estimators,
            learning_rate=lgb_learning_rate,
            num_leaves=15,
            log_transform=False,
            clip_non_negative=False,
            feature_cols=HybridProphetLGBMModel.ORGANIC_FEATURE_COLS,
            random_state=random_state,
        )

    def fit(self, df: pd.DataFrame, target_col: str = "demand") -> "SingleGameHybridModel":
        fit_df = df.copy()
        fit_df["date"] = pd.to_datetime(fit_df["date"])

        if "appid" in fit_df.columns:
            self.appid = int(fit_df["appid"].iloc[-1])
        if "game_name" in fit_df.columns:
            self.game_name = str(fit_df["game_name"].iloc[-1])
        if "peak_ratio" in fit_df.columns:
            self.peak_ratio = float(fit_df["peak_ratio"].iloc[-1])

        # Baseline scale (rolling 30d or recent baseline)
        if "sales_rolling_30" in fit_df.columns and fit_df["sales_rolling_30"].notna().any():
            scale = fit_df["sales_rolling_30"].clip(lower=10.0)
        else:
            scale = pd.Series(fit_df[target_col].median(), index=fit_df.index).clip(lower=10.0)

        self.base_rolling_30 = float(scale.iloc[-1])
        fit_df["_scale"] = scale
        fit_df["_norm_demand"] = fit_df[target_col] / fit_df["_scale"]

        is_sale = (
            fit_df["is_steam_sale"] == 1
            if "is_steam_sale" in fit_df.columns
            else pd.Series(False, index=fit_df.index)
        )
        non_sale = ~is_sale if (~is_sale).any() else pd.Series(True, index=fit_df.index)

        # 1. Fit game-specific Prophet model on clean unpromoted normalized series
        p_df = pd.DataFrame(
            {
                "ds": fit_df.loc[non_sale, "date"].dt.tz_localize(None),
                "y": np.log(fit_df.loc[non_sale, "_norm_demand"].clip(lower=0.01).astype(float)),
            }
        )
        self.prophet_model.fit(p_df)

        # In-sample predictions
        p_in = pd.DataFrame({"ds": fit_df["date"].dt.tz_localize(None)})
        p_forecast = self.prophet_model.predict(p_in)
        norm_base = np.clip(np.exp(p_forecast["yhat"].values), 0.0, None)
        prophet_base = norm_base * fit_df["_scale"].values

        # 2. Fit game-specific LightGBM model on organic log residuals
        X_org = HybridProphetLGBMModel._prepare_organic_features_df(fit_df, scale.values)
        log_org_ratio = np.log(
            np.clip(fit_df[target_col].values / np.maximum(prophet_base, 10.0), 0.05, 20.0)
        )
        self.lgb_organic_model.fit(X_org.loc[non_sale], log_org_ratio[non_sale])

        # 3. Fit uplift model if this game had sales
        if is_sale.any():
            X_up = HybridProphetLGBMModel._prepare_uplift_features_df(fit_df)
            y_actual = fit_df[target_col].values
            rel_uplift = np.clip(
                (y_actual[is_sale] / np.maximum(prophet_base[is_sale], 10.0)) - 1.0,
                0.0,
                10.0,
            )
            self.uplift_model.fit(X_up[is_sale], rel_uplift)
            self.has_empirical_sales = True
            self.empirical_mean_lift = float(rel_uplift.mean())

        return self

    def predict_components(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        if "sales_rolling_30" in df.columns and df["sales_rolling_30"].notna().any():
            scale = df["sales_rolling_30"].clip(lower=10.0).values
        elif "sales_lag_1" in df.columns and df["sales_lag_1"].notna().any():
            scale = df["sales_lag_1"].clip(lower=10.0).values
        else:
            scale = np.full(len(df), self.base_rolling_30)

        prophet_data = pd.DataFrame({"ds": pd.to_datetime(df["date"]).dt.tz_localize(None)})
        prophet_forecast = self.prophet_model.predict(prophet_data)
        norm_base = np.clip(np.exp(prophet_forecast["yhat"].values), 0.0, None)
        prophet_base = norm_base * scale

        X_org = HybridProphetLGBMModel._prepare_organic_features_df(df, scale)
        r_org = self.lgb_organic_model.predict(X_org)
        organic_demand = np.clip(prophet_base * np.exp(r_org), 0.0, None)

        # Uplift calculation
        is_sale = (
            df["is_steam_sale"] == 1
            if "is_steam_sale" in df.columns
            else np.zeros(len(df), dtype=bool)
        )
        if isinstance(is_sale, pd.Series):
            is_sale = is_sale.values

        if np.any(is_sale):
            X_up = HybridProphetLGBMModel._prepare_uplift_features_df(df)
            up_model = self.uplift_model if self.has_empirical_sales else self.fallback_uplift_model
            if up_model is not None and hasattr(up_model, "predict"):
                rel_lift = np.maximum(0.0, up_model.predict(X_up))
            else:
                disc = (
                    df["discount_depth"].values
                    if "discount_depth" in df.columns
                    else np.zeros(len(df))
                )
                rel_lift = np.maximum(0.0, (1.0 - disc) ** (-0.85) - 1.0)
            uplift_mult = np.where(is_sale, 1.0 + rel_lift, 1.0)
            combined = organic_demand * uplift_mult
        else:
            combined = organic_demand
            uplift_mult = np.ones(len(df))

        yearly_comp = prophet_forecast["yearly"].values if "yearly" in prophet_forecast else np.zeros(len(df))
        weekly_comp = prophet_forecast["weekly"].values if "weekly" in prophet_forecast else np.zeros(len(df))
        trend_comp = prophet_forecast["trend"].values if "trend" in prophet_forecast else prophet_base

        return {
            "combined": combined,
            "prophet_base": prophet_base,
            "organic_demand": organic_demand,
            "uplift_factor": uplift_mult,
            "lgbm_residual": organic_demand - prophet_base,
            "trend": trend_comp,
            "weekly": weekly_comp,
            "yearly": yearly_comp,
            "yhat_lower": np.clip(combined * 0.85, 0.0, None),
            "yhat_upper": combined * 1.15,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_components(df)["combined"]


class HybridProphetLGBMModel:
    """Hybrid Architecture supporting Per-Game Dedicated Models with Global Fallback."""

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
        changepoint_prior_scale: float = 0.01,
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        lgb_n_estimators: int = 150,
        lgb_learning_rate: float = 0.03,
        random_state: int = 42,
    ) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.lgb_n_estimators = lgb_n_estimators
        self.lgb_learning_rate = lgb_learning_rate
        self.random_state = random_state

        # Multi-game repository
        self.game_models: dict[int, SingleGameHybridModel] = {}

        # Global fallback Prophet & LightGBM
        self.prophet_model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale,
        )
        self.lgb_organic_model = LightGBMDemandModel(
            n_estimators=lgb_n_estimators,
            learning_rate=lgb_learning_rate,
            num_leaves=15,
            log_transform=False,
            clip_non_negative=False,
            feature_cols=self.ORGANIC_FEATURE_COLS,
            random_state=random_state,
        )
        self.uplift_model = Ridge(alpha=10.0, positive=True, random_state=random_state)

    @staticmethod
    def _prepare_organic_features_df(
        df: pd.DataFrame, scale: pd.Series | np.ndarray
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
            X[f"norm_rolling_std_{w}"] = (
                df[col_std].values / scale_safe if col_std in df.columns else 0.0
            )

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

    def _prepare_organic_features(
        self, df: pd.DataFrame, scale: pd.Series | np.ndarray
    ) -> pd.DataFrame:
        return self._prepare_organic_features_df(df, scale)

    @staticmethod
    def _prepare_uplift_features_df(df: pd.DataFrame) -> pd.DataFrame:
        """Create promo-structural features with continuous economic elasticity."""
        X = pd.DataFrame(index=df.index)
        disc = (
            df["discount_depth"].clip(0.0, 0.95)
            if "discount_depth" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        exposure = (
            df["storefront_exposure"]
            if "storefront_exposure" in df.columns
            else pd.Series(1.0, index=df.index)
        )
        weekend = (
            df["is_weekend"] if "is_weekend" in df.columns else pd.Series(0, index=df.index)
        )

        elast_lift = (1.0 - disc).pow(-DEFAULT_ELASTICITY_EPSILON) - 1.0
        weekend_factor = 1.0 + WEEKEND_BOOST * weekend
        X["base_elasticity"] = elast_lift * exposure * weekend_factor
        X["discount_depth"] = disc
        X["storefront_exposure"] = exposure
        X["is_weekend"] = weekend
        X["is_major_seasonal_sale"] = (
            df["is_major_seasonal_sale"].values
            if "is_major_seasonal_sale" in df.columns
            else 0
        )
        X["sale_progress"] = (
            df["sale_progress"].values if "sale_progress" in df.columns else 0.0
        )
        X["days_until_sale_end"] = (
            df["days_until_sale_end"].values
            if "days_until_sale_end" in df.columns
            else 0.0
        )

        for col in ["base_price", "is_multiplayer", "is_coop", "log_positive_reviews"]:
            X[col] = df[col].values if col in df.columns else 0.0

        return X

    def _prepare_uplift_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._prepare_uplift_features_df(df)

    def get_game_model(self, appid: int) -> Any:
        """Retrieve dedicated game model or fall back to global."""
        return self.game_models.get(int(appid), self)

    def fit(
        self, df: pd.DataFrame, target_col: str = "demand"
    ) -> "HybridProphetLGBMModel":
        """Fit game-specific hybrid models for each title alongside global fallback."""
        fit_df = df.copy()
        fit_df["date"] = pd.to_datetime(fit_df["date"])

        # 1. Fit global baseline scale and models
        if "sales_rolling_30" in fit_df.columns and fit_df["sales_rolling_30"].notna().any():
            scale = fit_df["sales_rolling_30"].clip(lower=10.0)
        else:
            scale = pd.Series(fit_df[target_col].median(), index=fit_df.index).clip(lower=10.0)

        fit_df["_scale"] = scale
        fit_df["_norm_demand"] = fit_df[target_col] / fit_df["_scale"]

        is_sale = (
            fit_df["is_steam_sale"] == 1
            if "is_steam_sale" in fit_df.columns
            else pd.Series(False, index=fit_df.index)
        )
        non_sale = ~is_sale if (~is_sale).any() else pd.Series(True, index=fit_df.index)

        macro_df = (
            fit_df.loc[non_sale].groupby("date")["_norm_demand"]
            .mean()
            .reset_index()
        )
        macro_prophet_data = pd.DataFrame()
        macro_prophet_data["ds"] = macro_df["date"].dt.tz_localize(None)
        macro_prophet_data["y"] = np.log(macro_df["_norm_demand"].clip(lower=0.01).astype(float))
        self.prophet_model.fit(macro_prophet_data)

        p_in = pd.DataFrame({"ds": fit_df["date"].dt.tz_localize(None)})
        p_forecast = self.prophet_model.predict(p_in)
        norm_base = np.clip(np.exp(p_forecast["yhat"].values), 0.0, None)
        prophet_base = norm_base * fit_df["_scale"].values

        X_org = self._prepare_organic_features(fit_df, scale.values)
        log_org_ratio = np.log(
            np.clip(fit_df[target_col].values / np.maximum(prophet_base, 10.0), 0.05, 20.0)
        )
        self.lgb_organic_model.fit(X_org.loc[non_sale], log_org_ratio[non_sale])

        # Uplift model fallback (trained on all historical sale observations)
        if is_sale.any():
            X_up = self._prepare_uplift_features(fit_df)
            y_actual = fit_df[target_col].values
            rel_uplift_target = np.clip(
                (y_actual[is_sale] / np.maximum(prophet_base[is_sale], 10.0)) - 1.0,
                0.0,
                10.0,
            )
            self.uplift_model.fit(X_up[is_sale], rel_uplift_target)

        # 2. Train dedicated per-game models inheriting the trained global uplift model
        if "appid" in fit_df.columns:
            for appid_val, group in fit_df.groupby("appid"):
                appid_int = int(appid_val)
                g_model = SingleGameHybridModel(
                    appid=appid_int,
                    game_name=str(group.get("game_name", pd.Series([""])).iloc[-1]),
                    changepoint_prior_scale=self.changepoint_prior_scale,
                    weekly_seasonality=self.weekly_seasonality,
                    lgb_n_estimators=self.lgb_n_estimators,
                    lgb_learning_rate=self.lgb_learning_rate,
                    random_state=self.random_state,
                    fallback_uplift_model=self.uplift_model,
                )
                g_model.fit(group, target_col=target_col)
                self.game_models[appid_int] = g_model

        return self

    def predict_components(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Generate decomposed components routing to per-game model if available."""
        if "appid" in df.columns:
            unique_appids = df["appid"].unique()
            if len(unique_appids) == 1 and int(unique_appids[0]) in self.game_models:
                return self.game_models[int(unique_appids[0])].predict_components(df)
            elif len(unique_appids) > 1 and all(int(a) in self.game_models for a in unique_appids):
                combined_arr = np.zeros(len(df))
                prophet_base_arr = np.zeros(len(df))
                organic_demand_arr = np.zeros(len(df))
                for appid_val, group in df.groupby("appid"):
                    g_res = self.game_models[int(appid_val)].predict_components(group)
                    idx = group.index
                    combined_arr[df.index.get_indexer(idx)] = g_res["combined"]
                    prophet_base_arr[df.index.get_indexer(idx)] = g_res["prophet_base"]
                    organic_demand_arr[df.index.get_indexer(idx)] = g_res["organic_demand"]
                return {
                    "combined": combined_arr,
                    "prophet_base": prophet_base_arr,
                    "organic_demand": organic_demand_arr,
                    "uplift_factor": np.ones(len(df)),
                    "lgbm_residual": combined_arr - prophet_base_arr,
                    "trend": prophet_base_arr,
                    "weekly": np.zeros(len(df)),
                    "yearly": np.zeros(len(df)),
                    "yhat_lower": np.clip(combined_arr * 0.85, 0.0, None),
                    "yhat_upper": combined_arr * 1.15,
                }

        # Global fallback
        if "sales_rolling_30" in df.columns and df["sales_rolling_30"].notna().any():
            scale = df["sales_rolling_30"].clip(lower=10.0).values
        elif "sales_lag_1" in df.columns and df["sales_lag_1"].notna().any():
            scale = df["sales_lag_1"].clip(lower=10.0).values
        else:
            scale = np.ones(len(df))

        prophet_data = pd.DataFrame({"ds": pd.to_datetime(df["date"]).dt.tz_localize(None)})
        prophet_forecast = self.prophet_model.predict(prophet_data)
        norm_base = np.clip(np.exp(prophet_forecast["yhat"].values), 0.0, None)
        prophet_base = norm_base * scale

        X_org = self._prepare_organic_features(df, scale)
        r_org = self.lgb_organic_model.predict(X_org)
        organic_demand = np.clip(prophet_base * np.exp(r_org), 0.0, None)

        # Uplift calculation
        is_sale = (
            df["is_steam_sale"] == 1
            if "is_steam_sale" in df.columns
            else np.zeros(len(df), dtype=bool)
        )
        if isinstance(is_sale, pd.Series):
            is_sale = is_sale.values

        if np.any(is_sale):
            X_up = HybridProphetLGBMModel._prepare_uplift_features_df(df)
            up_model = self.uplift_model if self.has_empirical_sales else self.fallback_uplift_model
            if up_model is not None and hasattr(up_model, "predict"):
                rel_lift = np.maximum(0.0, up_model.predict(X_up))
            else:
                disc = (
                    df["discount_depth"].values
                    if "discount_depth" in df.columns
                    else np.zeros(len(df))
                )
                rel_lift = np.maximum(0.0, (1.0 - disc) ** (-0.85) - 1.0)
            uplift_mult = np.where(is_sale, 1.0 + rel_lift, 1.0)
            combined = organic_demand * uplift_mult
        else:
            combined = organic_demand
            uplift_mult = np.ones(len(df))

        yearly_comp = prophet_forecast["yearly"].values if "yearly" in prophet_forecast else np.zeros(len(df))
        weekly_comp = prophet_forecast["weekly"].values if "weekly" in prophet_forecast else np.zeros(len(df))
        trend_comp = prophet_forecast["trend"].values if "trend" in prophet_forecast else prophet_base

        return {
            "combined": combined,
            "prophet_base": prophet_base,
            "organic_demand": organic_demand,
            "uplift_factor": uplift_mult,
            "lgbm_residual": organic_demand - prophet_base,
            "trend": trend_comp,
            "weekly": weekly_comp,
            "yearly": yearly_comp,
            "yhat_lower": np.clip(combined * 0.85, 0.0, None),
            "yhat_upper": combined * 1.15,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_components(df)["combined"]
