from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_PATH = Path("data/processed/payment_anomaly_features.csv")
OUTPUT_PATH = Path("data/processed/payment_anomaly_scored.csv")


NUMERIC_FEATURES = [
    "amount",
    "counterparty_risk_score",
    "historical_failure_count",
    "risk_adjusted_amount",
    "amount_log",
    "settlement_speed_days",
    "amount_z_score",
    "risk_adjusted_amount_z_score",
    "amount_percentile_rank",
    "risk_score_percentile_rank",
    "country_frequency",
    "currency_frequency",
]


CATEGORICAL_FEATURES = [
    "currency",
    "country",
    "payment_type",
    "channel",
    "settlement_window",
    "amount_bucket",
    "risk_band",
    "prior_failure_band",
]


BOOLEAN_FEATURES = [
    "is_weekend",
    "is_high_value",
    "has_prior_failures",
    "is_large_international",
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


DISPLAY_COLUMNS = [
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
    "isolation_forest_anomaly_score",
    "is_isolation_forest_anomaly",
    "isolation_forest_anomaly_rank",
    "anomaly_reasons",
    "anomaly_source",
    "recommended_action",
]


def load_dataset():
    df = pd.read_csv(INPUT_PATH)

    print("Loaded anomaly feature dataset")
    print("------------------------------")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Rule-based anomaly rate: {df['is_rule_based_anomaly'].mean():.2%}")

    return df


def build_isolation_forest_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
        ]
    )

    model = IsolationForest(
        n_estimators=300,
        contamination=0.04,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_and_score_anomalies(df):
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

    X = df[feature_columns]

    model = build_isolation_forest_pipeline()

    print("\nTraining Isolation Forest anomaly model...")
    model.fit(X)

    raw_predictions = model.predict(X)
    anomaly_scores = model.decision_function(X)

    scored_df = df.copy()

    # IsolationForest returns -1 for anomalies and 1 for normal records.
    scored_df["is_isolation_forest_anomaly"] = raw_predictions == -1

    # Lower decision_function score means more anomalous.
    scored_df["isolation_forest_anomaly_score"] = anomaly_scores

    scored_df["isolation_forest_anomaly_rank"] = scored_df[
        "isolation_forest_anomaly_score"
    ].rank(method="first", ascending=True)

    print("Model training and scoring completed")

    return scored_df, model


def compare_rule_and_model_anomalies(scored_df):
    print("\nRule-Based vs Isolation Forest Comparison")
    print("----------------------------------------")

    comparison = pd.crosstab(
        scored_df["is_rule_based_anomaly"],
        scored_df["is_isolation_forest_anomaly"],
        rownames=["Rule-based anomaly"],
        colnames=["Isolation Forest anomaly"],
    )

    print(comparison)

    rule_count = scored_df["is_rule_based_anomaly"].sum()
    model_count = scored_df["is_isolation_forest_anomaly"].sum()

    overlap_count = (
        scored_df["is_rule_based_anomaly"]
        & scored_df["is_isolation_forest_anomaly"]
    ).sum()

    scored_df["is_combined_anomaly"] = (
        scored_df["is_rule_based_anomaly"]
        | scored_df["is_isolation_forest_anomaly"]
    )

    print("\nSummary:")
    print(f"Rule-based anomalies: {rule_count}")
    print(f"Isolation Forest anomalies: {model_count}")
    print(f"Overlap anomalies: {overlap_count}")
    print(f"Combined anomalies: {scored_df['is_combined_anomaly'].sum()}")
    print(f"Combined anomaly rate: {scored_df['is_combined_anomaly'].mean():.2%}")

    if rule_count > 0:
        print(f"Rule anomaly coverage by model: {overlap_count / rule_count:.2%}")

    if model_count > 0:
        print(f"Model anomaly overlap with rules: {overlap_count / model_count:.2%}")


def add_anomaly_source_and_action(scored_df):
    scored_df = scored_df.copy()

    def identify_source(row):
        if row["is_rule_based_anomaly"] and row["is_isolation_forest_anomaly"]:
            return "RULE_AND_MODEL"
        if row["is_rule_based_anomaly"]:
            return "RULE_ONLY"
        if row["is_isolation_forest_anomaly"]:
            return "MODEL_ONLY"
        return "NORMAL"

    def recommend_action(source):
        if source == "RULE_AND_MODEL":
            return "High-priority review"
        if source == "RULE_ONLY":
            return "Review known risk pattern"
        if source == "MODEL_ONLY":
            return "Investigate unusual pattern"
        return "No anomaly review needed"

    scored_df["anomaly_source"] = scored_df.apply(identify_source, axis=1)
    scored_df["recommended_action"] = scored_df["anomaly_source"].apply(
        recommend_action
    )

    return scored_df

def print_top_model_anomalies(scored_df):
    print("\nTop 15 Isolation Forest Anomalies")
    print("---------------------------------")

    top_anomalies = scored_df.sort_values(
        "isolation_forest_anomaly_score",
        ascending=True,
    ).head(15)

    print(top_anomalies[DISPLAY_COLUMNS].to_string(index=False))


def print_model_only_anomalies(scored_df):
    print("\nTop Model-Only Anomalies")
    print("------------------------")

    model_only = scored_df[
        (scored_df["is_isolation_forest_anomaly"])
        & (~scored_df["is_rule_based_anomaly"])
    ].sort_values("isolation_forest_anomaly_score", ascending=True)

    if len(model_only) == 0:
        print("No model-only anomalies found.")
        return

    print(model_only[DISPLAY_COLUMNS].head(10).to_string(index=False))


def print_rule_only_anomalies(scored_df):
    print("\nTop Rule-Only Anomalies")
    print("-----------------------")

    rule_only = scored_df[
        (scored_df["is_rule_based_anomaly"])
        & (~scored_df["is_isolation_forest_anomaly"])
    ].sort_values("rule_anomaly_score", ascending=False)

    if len(rule_only) == 0:
        print("No rule-only anomalies found.")
        return

    print(rule_only[DISPLAY_COLUMNS].head(10).to_string(index=False))


def save_scored_dataset(scored_df):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved scored anomaly dataset to: {OUTPUT_PATH}")


def print_business_interpretation():
    print("\nBusiness Interpretation")
    print("-----------------------")
    print(
        "Rule-based anomalies capture known business patterns such as extreme amounts, "
        "large FILE payments, high-risk SWIFT payments, and prior-failure high-value payments."
    )
    print(
        "Isolation Forest may detect unusual combinations that are not covered by simple rules."
    )
    print(
        "The most useful anomaly workflow combines both: rules for known risk patterns "
        "and model-based detection for unusual patterns."
    )


def main():
    df = load_dataset()

    scored_df, model = train_and_score_anomalies(df)

    scored_df = add_anomaly_source_and_action(scored_df)
    compare_rule_and_model_anomalies(scored_df)
    print_top_model_anomalies(scored_df)
    print_model_only_anomalies(scored_df)
    print_rule_only_anomalies(scored_df)
    save_scored_dataset(scored_df)
    print_business_interpretation()


if __name__ == "__main__":
    main()
