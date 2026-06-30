import pandas as pd

from fastapi import FastAPI
from src.api.model_loader import (
    get_cash_forecast_artifact,
    get_payment_anomaly_artifact,
    get_payment_failure_model,
)
from src.api.schemas import (
    CashForecastRequest,
    CashForecastResponse,
    ModelInfo,
    ModelsResponse,
    PaymentAnomalyRequest,
    PaymentAnomalyResponse,
    PaymentFailureRequest,
    PaymentFailureResponse,
)
from src.forecasting.predict_cash_forecast import predict_next_day_cash_amounts
from src.anomaly.predict_anomaly import (
    add_anomaly_features,
    add_base_features,
    add_rule_based_anomaly_features,
    predict_anomalies,
)
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
        "version": "0.1.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "models_url": "/models",
        "prediction_endpoints": [
            "/predict/payment-failure",
            "/predict/cash-forecast",
            "/predict/payment-anomaly",
        ],
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "payments-intelligence-platform-api",
        "version": "0.1.0",
    }

@app.get("/models", response_model=ModelsResponse)
def list_models():
    return ModelsResponse(
        available_models=[
            ModelInfo(
                name="payment_failure_classifier",
                endpoint="/predict/payment-failure",
                model_type="classification",
                description=(
                    "Predicts payment failure risk and recommends operational action."
                ),
                status="available",
            ),
            ModelInfo(
                name="cash_forecast_model",
                endpoint="/predict/cash-forecast",
                model_type="regression",
                description=(
                    "Forecasts next-day total payment amount from daily cash activity."
                ),
                status="available",
            ),
            ModelInfo(
                name="payment_anomaly_detector",
                endpoint="/predict/payment-anomaly",
                model_type="anomaly_detection",
                description=(
                    "Detects unusual payment patterns using business rules "
                    "and Isolation Forest."
                ),
                status="available",
            ),
        ]
    )


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

@app.post("/predict/payment-anomaly", response_model=PaymentAnomalyResponse)
def predict_payment_anomaly(request: PaymentAnomalyRequest):
    artifact = get_payment_anomaly_artifact()

    payment_df = pd.DataFrame([request.model_dump()])

    payment_df = add_base_features(payment_df)
    payment_df = add_anomaly_features(payment_df)
    payment_df = add_rule_based_anomaly_features(payment_df)

    results_df = predict_anomalies(
        artifact=artifact,
        payments_df=payment_df,
    )

    result = results_df.iloc[0]

    return PaymentAnomalyResponse(
        payment_id=result["payment_id"],
        is_model_anomaly=bool(result["is_model_anomaly"]),
        is_rule_based_anomaly=bool(result["is_rule_based_anomaly"]),
        anomaly_score=float(result["anomaly_score"]),
        anomaly_band=result["anomaly_band"],
        anomaly_source=result["anomaly_source"],
        review_priority=result["review_priority"],
        anomaly_reasons=result["anomaly_reasons"],
        recommended_action=result["recommended_action"],
    )