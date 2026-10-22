"""LightGBM tree-based regressor conforming to course Listing 5."""

import lightgbm as lgb
import numpy as np
import pandas as pd


class LightGBMDemandModel:
    """M3 LightGBM model for nonlinear demand dynamics and promotional interactions."""

    DEFAULT_FEATURES = [
        "sales_lag_1",
        "sales_lag_2",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_30",
        "sales_rolling_3",
        "sales_rolling_std_3",
        "sales_rolling_7",
        "sales_rolling_std_7",
        "sales_rolling_14",
        "sales_rolling_std_14",
        "sales_rolling_30",
        "sales_rolling_std_30",
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
        "is_steam_sale",
        "is_major_seasonal_sale",
        "sale_day_index",
        "sale_progress",
        "days_until_sale_end",
        "days_since_last_sale",
        "storefront_exposure",
        "discount_depth",
        "effective_price",
        "post_sale_decay_7d",
        "promo_x_weekend",
        "promo_x_discount",
        "base_price",
        "is_multiplayer",
        "is_coop",
        "log_positive_reviews",
    ]

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        random_state: int = 42,
        log_transform: bool = True,
        clip_non_negative: bool = True,
        feature_cols: list[str] | None = None,
        monotone_constraints: dict | str | None = None,
    ) -> None:
        self.log_transform = log_transform
        self.clip_non_negative = clip_non_negative
        self.feature_cols = feature_cols or self.DEFAULT_FEATURES
        self.monotone_constraints = monotone_constraints
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            monotone_constraints=monotone_constraints,
            random_state=random_state,
            verbosity=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "LightGBMDemandModel":
        available_cols = [c for c in self.feature_cols if c in X.columns]
        y_train = np.log1p(y) if self.log_transform else y
        self.model.fit(X[available_cols], y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        available_cols = [c for c in self.feature_cols if c in X.columns]
        preds = self.model.predict(X[available_cols])
        if self.log_transform:
            preds = np.expm1(preds)
        if self.clip_non_negative:
            preds = np.clip(preds, 0.0, None)
        return preds

    @property
    def feature_importances_(self) -> pd.DataFrame:
        available_cols = [
            c
            for c in self.feature_cols
            if hasattr(self.model, "feature_name_") and c in self.model.feature_name_
        ]
        return pd.DataFrame(
            {
                "feature": available_cols,
                "importance": self.model.feature_importances_,
            }
        ).sort_values(by="importance", ascending=False)
