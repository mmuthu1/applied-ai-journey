import pandas as pd

from fastapi import FastAPI
from src.api.model_loader import get_payment_failure_model
from src.api.schemas import PaymentFailureRequest, PaymentFailureResponse
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
