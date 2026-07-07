from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/payments_features.csv")
OUTPUT_PATH = Path("data/processed/payments_failure_features_v2.csv")

TARGET_COLUMN = "is_failed"


def load_features():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_PATH}. "
            "Run python -m src.data.run_pipeline first."
        )

    df = pd.read_csv(INPUT_PATH)

    print("Loaded base feature dataset")
    print("---------------------------")
    print(f"Path: {INPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Failure rate: {df[TARGET_COLUMN].mean():.2%}")

    return df


def add_group_failure_rate(df, group_columns, output_column):
    group_failure_rates = (
        df.groupby(group_columns)[TARGET_COLUMN]
        .mean()
        .reset_index()
        .rename(columns={TARGET_COLUMN: output_column})
    )

    return df.merge(group_failure_rates, on=group_columns, how="left")


def add_failure_rate_features(df):
    df = add_group_failure_rate(
        df=df,
        group_columns=["payment_type"],
        output_column="payment_type_failure_rate",
    )

    df = add_group_failure_rate(
        df=df,
        group_columns=["channel"],
        output_column="channel_failure_rate",
    )

    df = add_group_failure_rate(
        df=df,
        group_columns=["country"],
        output_column="country_failure_rate",
    )

    df = add_group_failure_rate(
        df=df,
        group_columns=["currency"],
        output_column="currency_failure_rate",
    )

    df = add_group_failure_rate(
        df=df,
        group_columns=["payment_type", "channel"],
        output_column="payment_type_channel_failure_rate",
    )

    return df


def add_interaction_features(df):
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


def validate_v2_features(df):
    required_columns = [
        "payment_type_failure_rate",
        "channel_failure_rate",
        "country_failure_rate",
        "currency_failure_rate",
        "payment_type_channel_failure_rate",
        "risk_payment_interaction",
        "amount_risk_interaction",
        "prior_failure_risk_interaction",
        "high_value_high_risk",
        "swift_file_payment",
        "international_high_risk",
        "large_prior_failure_payment",
        "log_risk_adjusted_amount",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing v2 feature columns: {missing_columns}")

    missing_values = df[required_columns].isna().sum()
    columns_with_missing_values = missing_values[missing_values > 0]

    if not columns_with_missing_values.empty:
        raise ValueError(
            "Missing values found in v2 features:\n"
            f"{columns_with_missing_values}"
        )

    print("\nV2 feature validation")
    print("---------------------")
    print("All required v2 features created successfully.")
    print("No missing values found in v2 feature columns.")


def save_v2_features(df):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved v2 feature dataset")
    print("------------------------")
    print(f"Path: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


def print_feature_summary(df):
    v2_columns = [
        "payment_type_failure_rate",
        "channel_failure_rate",
        "country_failure_rate",
        "currency_failure_rate",
        "payment_type_channel_failure_rate",
        "risk_payment_interaction",
        "amount_risk_interaction",
        "prior_failure_risk_interaction",
        "high_value_high_risk",
        "swift_file_payment",
        "international_high_risk",
        "large_prior_failure_payment",
        "log_risk_adjusted_amount",
    ]

    print("\nV2 feature summary")
    print("------------------")
    print(df[v2_columns].head().to_string(index=False))

    print("\nV2 boolean feature rates")
    print("------------------------")
    boolean_columns = [
        "high_value_high_risk",
        "swift_file_payment",
        "international_high_risk",
        "large_prior_failure_payment",
    ]

    for column in boolean_columns:
        print(f"{column}: {df[column].mean():.2%}")


def print_leakage_note():
    print("\nLeakage note")
    print("------------")
    print(
        "This script creates dataset-level historical failure-rate features "
        "for learning and portfolio experimentation."
    )
    print(
        "In a production system, these features must be computed using only "
        "past data available before prediction time."
    )


def main():
    df = load_features()

    df = add_failure_rate_features(df)
    df = add_interaction_features(df)

    validate_v2_features(df)
    print_feature_summary(df)
    print_leakage_note()
    save_v2_features(df)


if __name__ == "__main__":
    main()
