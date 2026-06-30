from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "payments-intelligence-platform-api"
    assert data["version"] == "0.1.0"


def test_models_endpoint():
    response = client.get("/models")

    assert response.status_code == 200

    data = response.json()

    assert "available_models" in data
    assert len(data["available_models"]) == 3

    model_names = [model["name"] for model in data["available_models"]]

    assert "payment_failure_classifier" in model_names
    assert "cash_forecast_model" in model_names
    assert "payment_anomaly_detector" in model_names


def test_payment_failure_prediction_endpoint():
    payload = {
        "payment_id": "NEW-P1002",
        "amount": 1200000,
        "currency": "GBP",
        "country": "GB",
        "payment_type": "SWIFT",
        "channel": "FILE",
        "counterparty_risk_score": 0.82,
        "historical_failure_count": 4,
        "settlement_window": "T_PLUS_2",
        "payment_year": 2024,
        "payment_month": 6,
        "payment_day": 28,
        "day_of_week": 4,
        "hour_of_day": 18,
        "is_weekend": False,
    }

    response = client.post("/predict/payment-failure", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["payment_id"] == "NEW-P1002"
    assert data["predicted_is_failed"] in [0, 1]
    assert 0 <= data["predicted_failure_probability"] <= 1
    assert data["prediction_risk_band"] in ["LOW", "MEDIUM", "HIGH"]
    assert "recommended_action" in data


def test_cash_forecast_prediction_endpoint():
    payload = {
        "forecast_date": "2024-06-28",
        "daily_payment_count": 30,
        "daily_total_amount": 3500000,
        "daily_average_amount": 116666.67,
        "daily_median_amount": 65000,
        "failed_payment_count": 2,
        "high_value_payment_count": 1,
        "unique_currency_count": 6,
        "unique_country_count": 7,
        "failed_payment_rate": 0.0666666667,
        "high_value_payment_rate": 0.0333333333,
        "day_of_week": 4,
        "month": 6,
        "day_of_month": 28,
        "is_weekend": False,
        "previous_day_total_amount": 3600000,
        "previous_day_payment_count": 29,
        "rolling_3_day_avg_amount": 3400000,
        "rolling_7_day_avg_amount": 3550000,
        "rolling_3_day_payment_count": 28,
        "rolling_7_day_payment_count": 30,
    }

    response = client.post("/predict/cash-forecast", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["forecast_date"] == "2024-06-28"
    assert data["predicted_next_day_total_amount"] > 0
    assert data["forecast_band"] in ["LOW", "MEDIUM", "HIGH"]
    assert "recommended_action" in data
    assert "forecast_vs_7_day_avg" in data
    assert "forecast_vs_7_day_avg_pct" in data


def test_payment_anomaly_prediction_endpoint():
    payload = {
        "payment_id": "NEW-A1002",
        "amount": 2250000,
        "currency": "GBP",
        "country": "GB",
        "payment_type": "SWIFT",
        "channel": "FILE",
        "counterparty_risk_score": 0.82,
        "historical_failure_count": 4,
        "settlement_window": "T_PLUS_2",
        "payment_date": "2024-06-29 16:30:00",
    }

    response = client.post("/predict/payment-anomaly", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["payment_id"] == "NEW-A1002"
    assert isinstance(data["is_model_anomaly"], bool)
    assert isinstance(data["is_rule_based_anomaly"], bool)
    assert data["anomaly_band"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["anomaly_source"] in [
        "RULE_AND_MODEL",
        "RULE_ONLY",
        "MODEL_ONLY",
        "NORMAL",
    ]
    assert data["review_priority"] in [
        "P1_HIGH",
        "P2_INVESTIGATE",
        "P3_KNOWN_RISK",
        "P4_NORMAL",
    ]
    assert "anomaly_reasons" in data
    assert "recommended_action" in data
