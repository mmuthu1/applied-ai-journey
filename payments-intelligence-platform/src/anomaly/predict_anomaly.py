from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("models/payment_anomaly_detector.pkl")


def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run python -m src.anomaly.save_anomaly_model first."
        )

    artifact = joblib.load(MODEL_PATH)

    print("Loaded payment anomaly detector artifact")
    print("----------------------------------------")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model type: {artifact['model_type']}")
    print(f"Feature count: {len(artifact['feature_columns'])}")
    print(f"Contamination: {artifact['contamination']}")

    return artifact


def create_sample_payments():
    payments = [
        {
            "payment_id": "NEW-A1001",
            "amount": 45000,
            "currency": "USD",
            "country": "US",
            "payment_type": "ACH",
            "channel": "API",
            "counterparty_risk_score": 0.18,
            "historical_failure_count": 0,
            "settlement_window": "NEXT_DAY",
            "payment_date": "2024-06-29 10:00:00",
        },
        {
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
        },
        {
            "payment_id": "NEW-A1003",
            "amount": 675000,
            "currency": "JPY",
            "country": "JP",
            "payment_type": "SWIFT",
            "channel": "ONLINE",
            "counterparty_risk_score": 0.42,
            "historical_failure_count": 0,
            "settlement_window": "SAME_DAY",
            "payment_date": "2024-06-30 09:15:00",
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


def add_anomaly_features(df):
    df = df.copy()

    # In a real production system, these reference statistics should come from training metadata.
    # For this portfolio project, we use the same approximate statistics from the synthetic dataset.
    amount_mean = 125237.80
    amount_std = 234627.00

    risk_adjusted_mean = 36310.00
    risk_adjusted_std = 76938.00

    df["amount_z_score"] = (df["amount"] - amount_mean) / amount_std
    df["is_extreme_amount"] = df["amount_z_score"].abs() >= 3

    df["amount_percentile_rank"] = df["amount"].rank(pct=True)
    df["is_top_1pct_amount"] = df["amount_percentile_rank"] >= 0.99

    df["risk_adjusted_amount_z_score"] = (
        (df["risk_adjusted_amount"] - risk_adjusted_mean) / risk_adjusted_std
    )

    df["is_extreme_risk_adjusted_amount"] = (
        df["risk_adjusted_amount_z_score"].abs() >= 3
    )

    df["risk_score_percentile_rank"] = df["counterparty_risk_score"].rank(pct=True)
    df["is_top_1pct_risk_score"] = df["risk_score_percentile_rank"] >= 0.99

    # Approximate category frequency values for sample scoring.
    # Later we can save these from training data into the model artifact.
    df["country_frequency"] = 0.14
    df["currency_frequency"] = 0.16
    df["is_rare_country"] = df["country_frequency"] < 0.10
    df["is_rare_currency"] = df["currency_frequency"] < 0.10

    df["is_high_risk_large_payment"] = (
        (df["amount"] > 500000)
        & (df["counterparty_risk_score"] >= 0.70)
    )

    df["is_large_file_payment"] = (
        (df["amount"] > 500000)
        & (df["channel"] == "FILE")
    )

    df["is_high_risk_swift_payment"] = (
        (df["payment_type"] == "SWIFT")
        & (df["counterparty_risk_score"] >= 0.70)
    )

    df["is_prior_failure_high_value"] = (
        (df["historical_failure_count"] >= 3)
        & (df["amount"] > 500000)
    )

    return df


def add_rule_based_anomaly_features(df):
    df = df.copy()

    rule_columns = [
        "is_extreme_amount",
        "is_top_1pct_amount",
        "is_extreme_risk_adjusted_amount",
        "is_top_1pct_risk_score",
        "is_high_risk_large_payment",
        "is_large_file_payment",
        "is_high_risk_swift_payment",
        "is_prior_failure_high_value",
        "is_rare_country",
        "is_rare_currency",
    ]

    df["rule_anomaly_score"] = df[rule_columns].sum(axis=1)
    df["is_rule_based_anomaly"] = df["rule_anomaly_score"] > 0

    def build_reasons(row):
        reasons = []

        for column in rule_columns:
            if row[column]:
                reasons.append(column)

        if len(reasons) == 0:
            return "NONE"

        return "; ".join(reasons)

    df["anomaly_reasons"] = df.apply(build_reasons, axis=1)

    return df


def assign_anomaly_band(anomaly_score):
    # Lower Isolation Forest decision score means more anomalous.
    if anomaly_score < -0.05:
        return "HIGH"
    if anomaly_score < 0.02:
        return "MEDIUM"
    return "LOW"

def identify_anomaly_source(row):
    if row["is_rule_based_anomaly"] and row["is_model_anomaly"]:
        return "RULE_AND_MODEL"
    if row["is_rule_based_anomaly"]:
        return "RULE_ONLY"
    if row["is_model_anomaly"]:
        return "MODEL_ONLY"
    return "NORMAL"

def assign_review_priority(anomaly_source):
    if anomaly_source == "RULE_AND_MODEL":
        return "P1_HIGH"
    if anomaly_source == "MODEL_ONLY":
        return "P2_INVESTIGATE"
    if anomaly_source == "RULE_ONLY":
        return "P3_KNOWN_RISK"
    return "P4_NORMAL"

def recommend_action(row):
    if row["anomaly_source"] == "RULE_AND_MODEL":
        return "High-priority anomaly review"
    if row["anomaly_source"] == "MODEL_ONLY":
        return "Investigate model-detected unusual pattern"
    if row["anomaly_source"] == "RULE_ONLY":
        return "Review known business-risk pattern"
    return "No anomaly review needed"


def predict_anomalies(artifact, payments_df):
    model_pipeline = artifact["model_pipeline"]
    feature_columns = artifact["feature_columns"]

    feature_df = payments_df[feature_columns]

    raw_predictions = model_pipeline.predict(feature_df)
    anomaly_scores = model_pipeline.decision_function(feature_df)

    results_df = payments_df.copy()
    results_df["raw_isolation_forest_prediction"] = raw_predictions
    results_df["is_model_anomaly"] = raw_predictions == -1
    results_df["anomaly_score"] = anomaly_scores
    results_df["anomaly_band"] = results_df["anomaly_score"].apply(
        assign_anomaly_band
    )
    results_df["anomaly_source"] = results_df.apply(identify_anomaly_source, axis=1)
    results_df["recommended_action"] = results_df.apply(recommend_action, axis=1)
    results_df["review_priority"] = results_df["anomaly_source"].apply(
        assign_review_priority
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
        "risk_adjusted_amount",
        "rule_anomaly_score",
        "is_rule_based_anomaly",
        "is_model_anomaly",
        "anomaly_score",
        "anomaly_band",
        "recommended_action",
        "anomaly_reasons",
        "review_priority",
        "anomaly_source",
    ]

    print("\nPayment Anomaly Predictions")
    print("---------------------------")
    print(results_df[display_columns].to_string(index=False))

    print("\nBusiness interpretation")
    print("-----------------------")

    for _, row in results_df.iterrows():
        print(
            f"{row['payment_id']}: {row['anomaly_band']} anomaly band. "
            f"Rule anomaly: {row['is_rule_based_anomaly']}. "
            f"Model anomaly: {row['is_model_anomaly']}. "
            f"Action: {row['recommended_action']}."
        )


def main():
    artifact = load_model_artifact()

    print("\nCreating sample payment records...")
    payments_df = create_sample_payments()

    print("Adding base payment features...")
    payments_df = add_base_features(payments_df)

    print("Adding anomaly features...")
    payments_df = add_anomaly_features(payments_df)

    print("Adding rule-based anomaly features...")
    payments_df = add_rule_based_anomaly_features(payments_df)

    print("Scoring anomaly risk...")
    results_df = predict_anomalies(artifact, payments_df)

    print_predictions(results_df)

    print("\nAnomaly note")
    print("------------")
    print(
        "This anomaly detector is experimental and was trained on synthetic data. "
        "It should be used for learning and portfolio demonstration, not production payment decisions."
    )


if __name__ == "__main__":
    main()
