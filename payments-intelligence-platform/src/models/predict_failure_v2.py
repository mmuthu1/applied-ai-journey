from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("models/payment_failure_classifier_v2.pkl")


def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run python -m src.models.save_failure_model_v2 first."
        )

    artifact = joblib.load(MODEL_PATH)

    print("Loaded payment failure v2 model artifact")
    print("----------------------------------------")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model version: {artifact['model_version']}")
    print(f"Recommended threshold: {artifact['recommended_threshold']}")
    print(f"Feature count: {len(artifact['feature_columns'])}")

    return artifact


def create_sample_payments():
    payments = [
        {
            "payment_id": "NEW-V2-1001",
            "payment_date": "2024-06-15 10:00:00",
            "amount": 750000,
            "currency": "USD",
            "country": "US",
            "payment_type": "WIRE",
            "channel": "API",
            "counterparty_risk_score": 0.25,
            "historical_failure_count": 0,
            "settlement_window": "SAME_DAY",
        },
        {
            "payment_id": "NEW-V2-1002",
            "payment_date": "2024-06-28 18:00:00",
            "amount": 1200000,
            "currency": "GBP",
            "country": "GB",
            "payment_type": "SWIFT",
            "channel": "FILE",
            "counterparty_risk_score": 0.82,
            "historical_failure_count": 4,
            "settlement_window": "T_PLUS_2",
        },
        {
            "payment_id": "NEW-V2-1003",
            "payment_date": "2024-06-16 09:00:00",
            "amount": 8500,
            "currency": "USD",
            "country": "US",
            "payment_type": "ACH",
            "channel": "MOBILE",
            "counterparty_risk_score": 0.10,
            "historical_failure_count": 0,
            "settlement_window": "NEXT_DAY",
        },
    ]

    return pd.DataFrame(payments)


def add_base_features(df):
    df = df.copy()

    df["payment_date"] = pd.to_datetime(df["payment_date"])

    df["payment_year"] = df["payment_date"].dt.year
    df["payment_month"] = df["payment_date"].dt.month
    df["payment_day"] = df["payment_date"].dt.day
    df["day_of_week"] = df["payment_date"].dt.dayofweek
    df["hour_of_day"] = df["payment_date"].dt.hour
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

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


def add_v2_features(df):
    df = df.copy()

    # Approximate historical failure-rate features from the synthetic dataset.
    # In production, these should come from historical training metadata only.
    payment_type_failure_rates = {
        "ACH": 0.04834,
        "RTP": 0.04000,
        "SWIFT": 0.090196,
        "WIRE": 0.04172,
    }

    channel_failure_rates = {
        "API": 0.060700,
        "BATCH": 0.042173,
        "FILE": 0.080986,
        "MOBILE": 0.040123,
        "ONLINE": 0.054386,
    }

    # Conservative fallback values for country/currency and payment-type/channel.
    default_failure_rate = 0.0562

    df["payment_type_failure_rate"] = (
        df["payment_type"].map(payment_type_failure_rates).fillna(default_failure_rate)
    )

    df["channel_failure_rate"] = (
        df["channel"].map(channel_failure_rates).fillna(default_failure_rate)
    )

    df["country_failure_rate"] = default_failure_rate
    df["currency_failure_rate"] = default_failure_rate

    df["payment_type_channel_failure_rate"] = (
        df["payment_type_failure_rate"] + df["channel_failure_rate"]
    ) / 2

    df["risk_payment_interaction"] = (
        df["counterparty_risk_score"] * df["historical_failure_count"]
    )

    df["amount_risk_interaction"] = (
        df["amount_log"] * df["counterparty_risk_score"]
    )

    df["prior_failure_risk_interaction"] = (
        df["historical_failure_count"] * df["counterparty_risk_score"]
    )

    df["high_value_high_risk"] = (
        (df["is_high_value"]) & (df["counterparty_risk_score"] >= 0.70)
    )

    df["swift_file_payment"] = (
        (df["payment_type"] == "SWIFT") & (df["channel"] == "FILE")
    )

    df["international_high_risk"] = (
        (df["country"] != "US") & (df["counterparty_risk_score"] >= 0.70)
    )

    df["large_prior_failure_payment"] = (
        (df["amount"] > 500000) & (df["historical_failure_count"] > 0)
    )

    df["log_risk_adjusted_amount"] = np.log1p(df["risk_adjusted_amount"])

    return df


def assign_risk_band(probability, threshold):
    if probability >= threshold:
        return "HIGH"
    if probability >= 0.30:
        return "MEDIUM"
    return "LOW"


def recommend_action(risk_band):
    if risk_band == "HIGH":
        return "Review before release"
    if risk_band == "MEDIUM":
        return "Monitor or queue for secondary review"
    return "Allow normal processing"


def predict_payment_failures_v2(artifact, payments_df):
    model_pipeline = artifact["model_pipeline"]
    feature_columns = artifact["feature_columns"]
    threshold = artifact["recommended_threshold"]

    feature_df = payments_df[feature_columns]

    predicted_probabilities = model_pipeline.predict_proba(feature_df)[:, 1]
    predicted_classes = predicted_probabilities >= threshold

    results_df = payments_df.copy()
    results_df["predicted_is_failed"] = predicted_classes.astype(int)
    results_df["predicted_failure_probability"] = predicted_probabilities
    results_df["prediction_risk_band"] = results_df[
        "predicted_failure_probability"
    ].apply(lambda probability: assign_risk_band(probability, threshold))

    results_df["recommended_action"] = results_df["prediction_risk_band"].apply(
        recommend_action
    )

    return results_df


def print_predictions(results_df):
    display_columns = [
        "payment_id",
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
        "recommended_action",
    ]

    display_df = results_df[display_columns].copy()
    display_df["predicted_failure_probability"] = display_df[
        "predicted_failure_probability"
    ].round(4)

    print("\nPayment Failure v2 Predictions")
    print("------------------------------")
    print(display_df.to_string(index=False))

    print("\nBusiness interpretation")
    print("-----------------------")

    for _, row in results_df.iterrows():
        print(
            f"{row['payment_id']}: "
            f"{row['prediction_risk_band']} risk "
            f"with {row['predicted_failure_probability']:.2%} predicted failure probability. "
            f"Action: {row['recommended_action']}."
        )


def main():
    artifact = load_model_artifact()

    print("\nCreating sample payments...")
    payments_df = create_sample_payments()

    print("Adding base features...")
    payments_df = add_base_features(payments_df)

    print("Adding v2 features...")
    payments_df = add_v2_features(payments_df)

    print("Scoring sample payments...")
    results_df = predict_payment_failures_v2(
        artifact=artifact,
        payments_df=payments_df,
    )

    print_predictions(results_df)

    print("\nV2 model note")
    print("-------------")
    print(
        "This v2 model uses improved feature engineering and a recommended "
        "operating threshold of 0.55. It is still experimental and trained "
        "on synthetic data."
    )


if __name__ == "__main__":
    main()
