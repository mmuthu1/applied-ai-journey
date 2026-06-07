import numpy as np
import pandas as pd


INPUT_PATH = "data/processed/payments_clean.csv"
OUTPUT_PATH = "data/processed/payments_features.csv"


LEAKAGE_COLUMNS = [
    "status",
    "failure_reason",
]


def create_date_features(df):
    df["payment_date"] = pd.to_datetime(df["payment_date"], errors="coerce")

    df["payment_year"] = df["payment_date"].dt.year
    df["payment_month"] = df["payment_date"].dt.month
    df["payment_day"] = df["payment_date"].dt.day
    df["day_of_week"] = df["payment_date"].dt.dayofweek
    df["hour_of_day"] = df["payment_date"].dt.hour
    df["is_weekend"] = df["day_of_week"].isin([5, 6])

    return df


def create_amount_features(df):
    df["amount_log"] = np.log1p(df["amount"])

    df["is_high_value"] = df["amount"] > 500000

    df["amount_bucket"] = pd.cut(
        df["amount"],
        bins=[0, 10000, 100000, 500000, float("inf")],
        labels=["LOW", "MEDIUM", "HIGH", "VERY_HIGH"],
        include_lowest=True,
    )

    return df


def create_risk_features(df):
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

    return df


def create_settlement_features(df):
    settlement_speed_map = {
        "SAME_DAY": 0,
        "NEXT_DAY": 1,
        "T_PLUS_2": 2,
    }

    df["settlement_speed_days"] = df["settlement_window"].map(settlement_speed_map)

    return df


def create_business_rule_features(df):
    df["is_large_international"] = (
        (df["amount"] > 500000) & (df["country"] != "US")
    )

    df["risk_adjusted_amount"] = df["amount"] * (df["counterparty_risk_score"])

    return df

def remove_leakage_columns(df):
    columns_to_drop = []

    for column in LEAKAGE_COLUMNS:
        if column in df.columns:
            columns_to_drop.append(column)

    return df.drop(columns=columns_to_drop)

def create_features(input_path, output_path):
    df = pd.read_csv(input_path)

    original_columns = list(df.columns)

    df = create_date_features(df)
    df = create_amount_features(df)
    df = create_risk_features(df)
    df = create_settlement_features(df)
    df = create_business_rule_features(df)
    df = remove_leakage_columns(df)

    df.to_csv(output_path, index=False)

    print("Feature Engineering Report")
    print("--------------------------")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Input record count: {len(df)}")
    print(f"Original column count: {len(original_columns)}")
    print(f"Final column count: {len(df.columns)}")

    print("\nOriginal columns:")
    print(original_columns)

    print("\nFinal columns:")
    print(list(df.columns))

    print("\nTarget column:")
    print("is_failed")

    print("\nLeakage columns removed:")
    print(LEAKAGE_COLUMNS)

    print("\nFailure rate:")
    print(f"{df['is_failed'].mean():.2%}")

    print("\nAmount bucket distribution:")
    print(df["amount_bucket"].value_counts())

    print("\nRisk band distribution:")
    print(df["risk_band"].value_counts())

    print("\nPrior failure band distribution:")
    print(df["prior_failure_band"].value_counts())

    print("\nIs high value distribution:")
    print(df["is_high_value"].value_counts())

    print("\nIs large international payment distribution:")
    print(df["is_large_international"].value_counts())

    print("\nSample engineered records:")
    print(df.head())

    return df


def main():
    create_features(INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
