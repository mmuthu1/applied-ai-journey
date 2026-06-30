from typing import List
from pydantic import BaseModel, Field


class PaymentFailureRequest(BaseModel):
    payment_id: str
    amount: float = Field(gt=0)
    currency: str
    country: str
    payment_type: str
    channel: str
    counterparty_risk_score: float = Field(ge=0, le=1)
    historical_failure_count: int = Field(ge=0)
    settlement_window: str
    payment_year: int
    payment_month: int = Field(ge=1, le=12)
    payment_day: int = Field(ge=1, le=31)
    day_of_week: int = Field(ge=0, le=6)
    hour_of_day: int = Field(ge=0, le=23)
    is_weekend: bool


class PaymentFailureResponse(BaseModel):
    payment_id: str
    predicted_is_failed: int
    predicted_failure_probability: float
    prediction_risk_band: str
    recommended_action: str


class CashForecastRequest(BaseModel):
    forecast_date: str
    daily_payment_count: int = Field(ge=0)
    daily_total_amount: float = Field(ge=0)
    daily_average_amount: float = Field(ge=0)
    daily_median_amount: float = Field(ge=0)
    failed_payment_count: int = Field(ge=0)
    high_value_payment_count: int = Field(ge=0)
    unique_currency_count: int = Field(ge=0)
    unique_country_count: int = Field(ge=0)
    failed_payment_rate: float = Field(ge=0, le=1)
    high_value_payment_rate: float = Field(ge=0, le=1)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    day_of_month: int = Field(ge=1, le=31)
    is_weekend: bool
    previous_day_total_amount: float = Field(ge=0)
    previous_day_payment_count: int = Field(ge=0)
    rolling_3_day_avg_amount: float = Field(ge=0)
    rolling_7_day_avg_amount: float = Field(ge=0)
    rolling_3_day_payment_count: float = Field(ge=0)
    rolling_7_day_payment_count: float = Field(ge=0)


class CashForecastResponse(BaseModel):
    forecast_date: str
    predicted_next_day_total_amount: float
    forecast_band: str
    recommended_action: str
    forecast_vs_7_day_avg: float
    forecast_vs_7_day_avg_pct: float


class PaymentAnomalyRequest(BaseModel):
    payment_id: str
    amount: float = Field(gt=0)
    currency: str
    country: str
    payment_type: str
    channel: str
    counterparty_risk_score: float = Field(ge=0, le=1)
    historical_failure_count: int = Field(ge=0)
    settlement_window: str
    payment_date: str


class PaymentAnomalyResponse(BaseModel):
    payment_id: str
    is_model_anomaly: bool
    is_rule_based_anomaly: bool
    anomaly_score: float
    anomaly_band: str
    anomaly_source: str
    review_priority: str
    anomaly_reasons: str
    recommended_action: str


class ModelInfo(BaseModel):
    name: str
    endpoint: str
    model_type: str
    description: str
    status: str


class ModelsResponse(BaseModel):
    available_models: List[ModelInfo]
