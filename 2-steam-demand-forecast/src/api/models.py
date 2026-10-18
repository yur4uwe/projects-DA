from datetime import date

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    appid: int = Field(default=367520, description="Steam AppID")
    horizon_days: int = Field(
        default=14, ge=1, le=60, description="Forecast horizon in days"
    )
    start_date: date | None = Field(
        default=None, description="Start date for simulation"
    )
    discount_depth: float = Field(
        default=0.33,
        ge=0.0,
        le=0.90,
        description="Planned discount percentage (0.0 to 0.9)",
    )
    sale_duration_days: int = Field(
        default=7, ge=1, le=30, description="Sale duration in days"
    )
    is_custom_event: bool = Field(
        default=False, description="Whether this is a custom publisher deal"
    )
    custom_event_name: str = Field(
        default="Custom Publisher Sale", description="Name of custom event"
    )
    model_engine: str = Field(default="hybrid", description="'hybrid' or 'lgbm'")


class ElasticityRequest(BaseModel):
    appid: int = Field(default=367520, description="Steam AppID")
    base_price: float = Field(default=14.99, gt=0, description="Base price in USD")
    start_date: date | None = Field(
        default=None, description="Start date for discount test"
    )
    sale_duration_days: int = Field(
        default=7, ge=1, le=30, description="Sale duration in days"
    )
    is_custom_event: bool = Field(default=False)
    test_discounts: list[float] = Field(default=[0.10, 0.20, 0.33, 0.50, 0.66, 0.75])
    model_engine: str = Field(default="hybrid")
