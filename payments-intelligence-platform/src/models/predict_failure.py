from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("models/payment_failure_classifier.pkl")


FEATURE_COLUMNS = [
    "amount",
    "counterparty_risk_score",
    "historical_failure_count",
    "payment_year",
    "payment_month",
    "payment_day",
    "day_of_week",
    "hour_of_day",
    "amount_log",
    "settlement_speed_days",
    "risk_adjusted_amount",
    "currency",
    "country",
    "payment_type",
    "channel",
    "settlement_window",
    "amount_bucket",
    "risk_band",
    "prior_failure_band",
    "is_weekend",
    "is_high_value",
    "has_prior_failures",
    "is_large_international",
]


def create_sample_payments():
    payments = [
        {
            "payment_id": "NEW-P1001",
            "amount": 750000,
            "currency": "USD",
            "country": "US",
            "payment_type": "WIRE",
            "channel": "API",
            "counterparty_risk_score": 0.25,
            "historical_failure_count": 0,
            "settlement_window": "SAME_DAY",
            "payment_year": 2024,
            "payment_month": 6,
            "payment_day": 15,
            "day_of_week": 2,
            "hour_of_day": 10,
            "is_weekend": False,
        },
        {
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
        },
        {
            "payment_id": "NEW-P1003",
            "amount": 8500,
            "currency": "USD",
            "country": "US",
            "payment_type": "ACH",
            "channel": "MOBILE",
            "counterparty_risk_score": 0.10,
            "historical_failure_count": 0,
            "settlement_window": "NEXT_DAY",
            "payment_year": 2024,
            "payment_month": 6,
            "payment_day": 16,
            "day_of_week": 6,
            "hour_of_day": 9,
            "is_weekend": True,
        },
    ]

    return pd.DataFrame(payments)


def add_engineered_features(df):
    df["amount_log"] = np.log1p(df["amount"])

    df["is_high_value"] = df["amount"] > 500000

    df["amount_bucket"] = pd.cut(
        df["amount"],
        bins=[0, 10000, 100000, 500000, float("inf")],
        labels=["LOW", "MEDIUM", "HIGH", "VERY_HIGH"],
        include_lowest=True,
    )

    df["risk_band"] = pd.cut(
        df["counterparty_risk_score"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"],
        include_lowest=True,
    )

    df["has_prior_failures"] = df["historical_failure_count"] > 0

    df["prior_failure_band"] = pd.cut(
        df["historical_failure_count"],
        bins=[-1, 0, 2, float("inf")],
        labels=["NONE", "LOW", "HIGH"],
    )

    settlement_speed_map = {
        "SAME_DAY": 0,
        "NEXT_DAY": 1,
        "T_PLUS_2": 2,
    }

    df["settlement_speed_days"] = df["settlement_window"].map(settlement_speed_map)

    df["is_large_international"] = (
        (df["amount"] > 500000) & (df["country"] != "US")
    )

    df["risk_adjusted_amount"] = (
        df["amount"] * df["counterparty_risk_score"]
    )

    return df


def assign_risk_band(probability):
    if probability >= 0.60:
        return "HIGH"
    if probability >= 0.30:
        return "MEDIUM"
    return "LOW"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run python -m src.models.save_failure_model first."
        )

    return joblib.load(MODEL_PATH)

def recommend_action(risk_band):
    if risk_band == "HIGH":
        return "Review before release"
    if risk_band == "MEDIUM":
        return "Monitor or queue for secondary review"
    return "Allow normal processing"

def predict_payment_failures(model, payments_df):
    feature_df = payments_df[FEATURE_COLUMNS]

    predicted_classes = model.predict(feature_df)
    predicted_probabilities = model.predict_proba(feature_df)[:, 1]

    results_df = payments_df.copy()
    results_df["predicted_is_failed"] = predicted_classes
    results_df["predicted_failure_probability"] = predicted_probabilities
    results_df["prediction_risk_band"] = results_df[
        "predicted_failure_probability"
    ].apply(assign_risk_band)

    results_df["recommended_action"] = results_df[
    "prediction_risk_band"
    ].apply(recommend_action)

    return results_df


def print_predictions(results_df):
    display_columns = [
        "amount",
        "currency",
        "country",
        "payment_type",
        "channel",
        "counterparty_risk_score",
        "historical_failure_count",
        "settlement_window",
        "predicted_is_failed",
        "predicted_failure_probability",
        "prediction_risk_band",
        "payment_id",
        "recommended_action",
    ]

    print("\nPayment Failure Predictions")
    print("---------------------------")
    print(results_df[display_columns].to_string(index=False))

    print("\nBusiness interpretation")
    print("-----------------------")

    for index, row in results_df.iterrows():
        print(
            f"Payment {index + 1}: "
            f"{row['prediction_risk_band']} risk "
            f"with {row['predicted_failure_probability']:.2%} predicted failure probability."
        )
    
    print("\nThreshold note")
    print("--------------")
    print("HIGH risk is assigned when predicted failure probability is >= 60%.")
    print("This threshold is based on Week 2 threshold tuning and is experimental.")

def main():
    print("Loading saved payment failure model...")
    model = load_model()

    print("Creating sample payment records...")
    payments_df = create_sample_payments()

    print("Adding engineered features...")
    payments_df = add_engineered_features(payments_df)

    print("Scoring payments...")
    results_df = predict_payment_failures(model, payments_df)

    print_predictions(results_df)


if __name__ == "__main__":
    main()
