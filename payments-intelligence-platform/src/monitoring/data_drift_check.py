from pathlib import Path

import pandas as pd


REFERENCE_DATA_PATH = Path("data/processed/payments_features.csv")
OUTPUT_PATH = Path("data/monitoring/data_drift_summary.csv")


NUMERIC_COLUMNS = [
    "amount",
    "counterparty_risk_score",
    "historical_failure_count",
    "risk_adjusted_amount",
]

CATEGORICAL_COLUMNS = [
    "payment_type",
    "channel",
    "country",
    "currency",
]


def load_reference_data():
    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference data not found: {REFERENCE_DATA_PATH}. "
            "Run python -m src.data.run_pipeline first."
        )

    df = pd.read_csv(REFERENCE_DATA_PATH)

    print("Loaded reference dataset")
    print("------------------------")
    print(f"Path: {REFERENCE_DATA_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


def create_simulated_new_data(reference_df):
    """
    Create a small simulated scoring dataset with intentional distribution shifts.

    This is only for monitoring demonstration.
    In production, this would be replaced by recent scored payments.
    """
    new_df = reference_df.sample(n=500, random_state=42).copy()

    # Simulate amount drift: new payment amounts are larger.
    new_df["amount"] = new_df["amount"] * 1.35

    # Simulate risk drift: counterparties are slightly riskier.
    new_df["counterparty_risk_score"] = (
        new_df["counterparty_risk_score"] + 0.08
    ).clip(0, 1)

    # Recalculate risk-adjusted amount after simulated changes.
    new_df["risk_adjusted_amount"] = (
        new_df["amount"] * new_df["counterparty_risk_score"]
    )

    # Simulate channel mix drift by increasing FILE volume.
    file_sample = new_df.sample(frac=0.18, random_state=99).index
    new_df.loc[file_sample, "channel"] = "FILE"

    # Simulate payment type drift by increasing SWIFT volume.
    swift_sample = new_df.sample(frac=0.15, random_state=100).index
    new_df.loc[swift_sample, "payment_type"] = "SWIFT"

    print("\nCreated simulated new scoring dataset")
    print("-------------------------------------")
    print(f"Rows: {len(new_df)}")
    print("Simulated shifts: higher amount, higher risk score, more FILE/SWIFT volume")

    return new_df


def calculate_numeric_drift(reference_df, new_df, column):
    reference_mean = reference_df[column].mean()
    new_mean = new_df[column].mean()

    reference_median = reference_df[column].median()
    new_median = new_df[column].median()

    if reference_mean == 0:
        mean_change_pct = 0
    else:
        mean_change_pct = ((new_mean - reference_mean) / reference_mean) * 100

    if abs(mean_change_pct) >= 25:
        drift_level = "HIGH"
    elif abs(mean_change_pct) >= 10:
        drift_level = "MEDIUM"
    else:
        drift_level = "LOW"

    return {
        "feature": column,
        "feature_type": "numeric",
        "reference_value": round(reference_mean, 4),
        "new_value": round(new_mean, 4),
        "comparison_metric": "mean",
        "change_pct": round(mean_change_pct, 2),
        "drift_level": drift_level,
        "reference_median": round(reference_median, 4),
        "new_median": round(new_median, 4),
    }


def get_top_category_share(df, column):
    value_counts = df[column].value_counts(normalize=True)

    if value_counts.empty:
        return None, 0

    top_category = value_counts.index[0]
    top_share = value_counts.iloc[0]

    return top_category, top_share


def calculate_categorical_drift(reference_df, new_df, column):
    reference_top_category, reference_top_share = get_top_category_share(
        reference_df,
        column,
    )
    new_top_category, new_top_share = get_top_category_share(
        new_df,
        column,
    )

    share_change_pct_points = (new_top_share - reference_top_share) * 100

    top_category_changed = reference_top_category != new_top_category

    if top_category_changed or abs(share_change_pct_points) >= 10:
        drift_level = "HIGH"
    elif abs(share_change_pct_points) >= 5:
        drift_level = "MEDIUM"
    else:
        drift_level = "LOW"

    return {
        "feature": column,
        "feature_type": "categorical",
        "reference_value": reference_top_category,
        "new_value": new_top_category,
        "comparison_metric": "top_category_share",
        "change_pct": round(share_change_pct_points, 2),
        "drift_level": drift_level,
        "reference_median": None,
        "new_median": None,
    }


def build_drift_summary(reference_df, new_df):
    drift_rows = []

    for column in NUMERIC_COLUMNS:
        drift_rows.append(
            calculate_numeric_drift(
                reference_df=reference_df,
                new_df=new_df,
                column=column,
            )
        )

    for column in CATEGORICAL_COLUMNS:
        drift_rows.append(
            calculate_categorical_drift(
                reference_df=reference_df,
                new_df=new_df,
                column=column,
            )
        )

    return pd.DataFrame(drift_rows)


def save_drift_summary(drift_df):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    drift_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSaved data drift summary")
    print("------------------------")
    print(f"Path: {OUTPUT_PATH}")
    print(f"Rows: {len(drift_df)}")


def print_drift_summary(drift_df):
    print("\nData drift summary")
    print("------------------")
    print(drift_df.to_string(index=False))

    print("\nDrift levels")
    print("------------")
    print(drift_df["drift_level"].value_counts().to_string())


def main():
    reference_df = load_reference_data()
    new_df = create_simulated_new_data(reference_df)

    drift_df = build_drift_summary(
        reference_df=reference_df,
        new_df=new_df,
    )

    print_drift_summary(drift_df)
    save_drift_summary(drift_df)

    print("\nMonitoring note")
    print("---------------")
    print(
        "This drift check uses a simulated new scoring dataset for portfolio "
        "demonstration. In production, the new dataset would come from recent "
        "scored payments or live prediction logs."
    )


if __name__ == "__main__":
    main()
