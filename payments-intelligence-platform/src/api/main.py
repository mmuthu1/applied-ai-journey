import pandas as pd

from fastapi import FastAPI
from src.api.model_loader import (
    get_cash_forecast_artifact,
    get_payment_failure_model,
)
from src.api.schemas import (
    CashForecastRequest,
    CashForecastResponse,
    PaymentFailureRequest,
    PaymentFailureResponse,
)
from src.forecasting.predict_cash_forecast import predict_next_day_cash_amounts
from src.models.predict_failure import (
    add_engineered_features,
    predict_payment_failures,
)


app = FastAPI(
    title="Payments Intelligence Platform API",
    description=(
        "API for payment failure prediction, cash forecasting, "
        "and payment anomaly detection."
    ),
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "Payments Intelligence Platform API",
        "status": "running",
        "available_endpoints": [
            "/health",
            "/docs",
        ],
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "payments-intelligence-platform-api",
        "version": "0.1.0",
    }

@app.post("/predict/payment-failure", response_model=PaymentFailureResponse)
def predict_payment_failure(request: PaymentFailureRequest):
    model = get_payment_failure_model()

    payment_df = pd.DataFrame([request.model_dump()])
    payment_df = add_engineered_features(payment_df)

    results_df = predict_payment_failures(model, payment_df)
    result = results_df.iloc[0]

    return PaymentFailureResponse(
        payment_id=result["payment_id"],
        predicted_is_failed=int(result["predicted_is_failed"]),
        predicted_failure_probability=float(
            result["predicted_failure_probability"]
        ),
        prediction_risk_band=result["prediction_risk_band"],
        recommended_action=result["recommended_action"],
    )

@app.post("/predict/cash-forecast", response_model=CashForecastResponse)
def predict_cash_forecast(request: CashForecastRequest):
    artifact = get_cash_forecast_artifact()

    cash_activity_df = pd.DataFrame([request.model_dump()])

    results_df = predict_next_day_cash_amounts(
        artifact=artifact,
        cash_activity_df=cash_activity_df,
    )

    result = results_df.iloc[0]

    return CashForecastResponse(
        forecast_date=result["forecast_date"],
        predicted_next_day_total_amount=float(
            result["predicted_next_day_total_amount"]
        ),
        forecast_band=result["forecast_band"],
        recommended_action=result["recommended_action"],
        forecast_vs_7_day_avg=float(result["forecast_vs_7_day_avg"]),
        forecast_vs_7_day_avg_pct=float(result["forecast_vs_7_day_avg_pct"]),
    )
