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
