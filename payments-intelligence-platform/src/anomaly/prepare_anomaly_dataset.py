from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/payments_features.csv")
OUTPUT_PATH = Path("data/processed/payment_anomaly_features.csv")


def load_payments():
    df = pd.read_csv(INPUT_PATH)
    df["payment_date"] = pd.to_datetime(df["payment_date"])

    print("Loaded payment features dataset")
    print("-------------------------------")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df['payment_date'].min()} to {df['payment_date'].max()}")

    return df


def add_amount_anomaly_features(df):
    df = df.copy()

    amount_mean = df["amount"].mean()
    amount_std = df["amount"].std()

    df["amount_z_score"] = (df["amount"] - amount_mean) / amount_std
    df["is_extreme_amount"] = df["amount_z_score"].abs() >= 3

    df["amount_percentile_rank"] = df["amount"].rank(pct=True)
    df["is_top_1pct_amount"] = df["amount_percentile_rank"] >= 0.99

    return df


def add_risk_anomaly_features(df):
    df = df.copy()

    risk_adjusted_mean = df["risk_adjusted_amount"].mean()
    risk_adjusted_std = df["risk_adjusted_amount"].std()

    df["risk_adjusted_amount_z_score"] = (
        (df["risk_adjusted_amount"] - risk_adjusted_mean) / risk_adjusted_std
    )

    df["is_extreme_risk_adjusted_amount"] = (
        df["risk_adjusted_amount_z_score"].abs() >= 3
    )

    df["risk_score_percentile_rank"] = df["counterparty_risk_score"].rank(pct=True)
    df["is_top_1pct_risk_score"] = df["risk_score_percentile_rank"] >= 0.99

    return df


def add_behavioral_anomaly_flags(df):
    df = df.copy()

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

def add_rare_category_flags(df):
    df = df.copy()

    country_rates = df["country"].value_counts(normalize=True)
    currency_rates = df["currency"].value_counts(normalize=True)

    df["country_frequency"] = df["country"].map(country_rates)
    df["currency_frequency"] = df["currency"].map(currency_rates)

    df["is_rare_country"] = df["country_frequency"] < 0.10
    df["is_rare_currency"] = df["currency_frequency"] < 0.10

    return df

def add_anomaly_reason_summary(df):
    df = df.copy()

    reason_columns = [
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

    def build_reasons(row):
        reasons = []

        for column in reason_columns:
            if row[column]:
                reasons.append(column)

        if len(reasons) == 0:
            return "NONE"

        return "; ".join(reasons)

    df["anomaly_reasons"] = df.apply(build_reasons, axis=1)

    return df

def create_rule_based_anomaly_label(df):
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

    return df


def print_anomaly_dataset_summary(df):
    print("\nPayment Anomaly Dataset Summary")
    print("-------------------------------")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nRule-based anomaly count:")
    print(df["is_rule_based_anomaly"].value_counts())

    print("\nRule-based anomaly rate:")
    print(f"{df['is_rule_based_anomaly'].mean():.2%}")

    print("\nTop rule anomaly score distribution:")
    print(df["rule_anomaly_score"].value_counts().sort_index())

    print("\nTop 10 rule-based anomalies:")
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
        "amount_z_score",
        "risk_adjusted_amount_z_score",
        "rule_anomaly_score",
        "is_rule_based_anomaly",
        "anomaly_reasons",
    ]

    top_anomalies = df.sort_values(
        ["rule_anomaly_score", "risk_adjusted_amount"],
        ascending=False,
    ).head(10)

    print(top_anomalies[display_columns].to_string(index=False))

    print("\nTop anomaly reasons:")
    reason_counts = (
        df[df["anomaly_reasons"] != "NONE"]["anomaly_reasons"]
        .value_counts()
        .head(10)
    )
    print(reason_counts)

def create_anomaly_dataset():
    df = load_payments()

    df = add_amount_anomaly_features(df)
    df = add_risk_anomaly_features(df)
    df = add_behavioral_anomaly_flags(df)
    df = add_rare_category_flags(df)
    df = create_rule_based_anomaly_label(df)
    df = add_anomaly_reason_summary(df)


    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved anomaly dataset to: {OUTPUT_PATH}")

    print_anomaly_dataset_summary(df)

    return df


def main():
    create_anomaly_dataset()


if __name__ == "__main__":
    main()
