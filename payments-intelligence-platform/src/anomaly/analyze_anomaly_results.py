from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/payment_anomaly_scored.csv")


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
    "is_isolation_forest_anomaly",
    "is_combined_anomaly",
    "isolation_forest_anomaly_score",
    "isolation_forest_anomaly_rank",
    "anomaly_source",
    "recommended_action",
    "anomaly_reasons",
    "review_priority",
]


def load_scored_anomalies():
    df = pd.read_csv(INPUT_PATH)

    print("Loaded scored anomaly dataset")
    print("-----------------------------")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


def print_anomaly_source_summary(df):
    print("\nAnomaly Source Summary")
    print("----------------------")
    print(df["anomaly_source"].value_counts())

    print("\nAnomaly Source Rate")
    print("-------------------")
    print((df["anomaly_source"].value_counts(normalize=True) * 100).round(2))


def print_recommended_action_summary(df):
    print("\nRecommended Action Summary")
    print("--------------------------")
    print(df["recommended_action"].value_counts())


def print_combined_anomaly_summary(df):
    print("\nCombined Anomaly Summary")
    print("------------------------")

    total_records = len(df)
    combined_count = df["is_combined_anomaly"].sum()
    rule_count = df["is_rule_based_anomaly"].sum()
    model_count = df["is_isolation_forest_anomaly"].sum()

    both_count = (
        df["is_rule_based_anomaly"]
        & df["is_isolation_forest_anomaly"]
    ).sum()

    print(f"Total records: {total_records}")
    print(f"Rule-based anomalies: {rule_count}")
    print(f"Isolation Forest anomalies: {model_count}")
    print(f"Rule and model overlap: {both_count}")
    print(f"Combined anomalies: {combined_count}")
    print(f"Combined anomaly rate: {combined_count / total_records:.2%}")


def print_top_anomaly_reasons(df):
    print("\nTop Anomaly Reasons")
    print("-------------------")

    anomaly_df = df[df["anomaly_reasons"] != "NONE"]

    if len(anomaly_df) == 0:
        print("No anomaly reasons found.")
        return

    print(anomaly_df["anomaly_reasons"].value_counts().head(15))


def print_high_priority_review_queue(df):
    print("\nHigh-Priority Review Queue")
    print("--------------------------")

    high_priority_df = df[
        df["anomaly_source"] == "RULE_AND_MODEL"
    ].sort_values(
        ["rule_anomaly_score", "isolation_forest_anomaly_rank"],
        ascending=[False, True],
    )

    if len(high_priority_df) == 0:
        print("No high-priority anomalies found.")
        return

    print(high_priority_df[DISPLAY_COLUMNS].head(15).to_string(index=False))


def print_model_only_review_queue(df):
    print("\nModel-Only Review Queue")
    print("-----------------------")

    model_only_df = df[
        df["anomaly_source"] == "MODEL_ONLY"
    ].sort_values(
        "isolation_forest_anomaly_rank",
        ascending=True,
    )

    if len(model_only_df) == 0:
        print("No model-only anomalies found.")
        return

    print(model_only_df[DISPLAY_COLUMNS].head(10).to_string(index=False))


def print_rule_only_review_queue(df):
    print("\nRule-Only Review Queue")
    print("----------------------")

    rule_only_df = df[
        df["anomaly_source"] == "RULE_ONLY"
    ].sort_values(
        ["rule_anomaly_score", "risk_adjusted_amount"],
        ascending=[False, False],
    )

    if len(rule_only_df) == 0:
        print("No rule-only anomalies found.")
        return

    print(rule_only_df[DISPLAY_COLUMNS].head(10).to_string(index=False))


def print_amount_summary_by_anomaly_source(df):
    print("\nAmount Summary by Anomaly Source")
    print("--------------------------------")

    summary = (
        df.groupby("anomaly_source")
        .agg(
            payment_count=("payment_id", "count"),
            average_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            max_amount=("amount", "max"),
            average_risk_score=("counterparty_risk_score", "mean"),
            average_risk_adjusted_amount=("risk_adjusted_amount", "mean"),
        )
        .sort_values("average_amount", ascending=False)
    )

    print(summary.round(2).to_string())


def print_failure_rate_by_anomaly_source(df):
    print("\nFailure Rate by Anomaly Source")
    print("------------------------------")

    if "is_failed" not in df.columns:
        print("is_failed column not available.")
        return

    summary = (
        df.groupby("anomaly_source")
        .agg(
            payment_count=("payment_id", "count"),
            failed_count=("is_failed", "sum"),
            failure_rate=("is_failed", "mean"),
        )
        .sort_values("failure_rate", ascending=False)
    )

    print(summary.to_string())


def print_business_interpretation(df):
    print("\nBusiness Interpretation")
    print("-----------------------")

    source_counts = df["anomaly_source"].value_counts()

    rule_and_model_count = source_counts.get("RULE_AND_MODEL", 0)
    model_only_count = source_counts.get("MODEL_ONLY", 0)
    rule_only_count = source_counts.get("RULE_ONLY", 0)

    print(
        f"{rule_and_model_count} payments were flagged by both business rules and the model. "
        "These should be treated as the highest-priority review queue."
    )

    print(
        f"{model_only_count} payments were flagged only by the model. "
        "These may represent unusual combinations not covered by existing rules."
    )

    print(
        f"{rule_only_count} payments were flagged only by rules. "
        "These represent known business risk patterns that the model did not consider unusual enough."
    )

    print(
        "A practical anomaly workflow should combine rules and model-based detection, "
        "then route payments based on anomaly source and severity."
    )


def print_recommendation():
    print("\nRecommendation")
    print("--------------")
    print(
        "Use RULE_AND_MODEL anomalies as the first review queue because both business logic "
        "and the unsupervised model agree they are unusual."
    )
    print(
        "Use MODEL_ONLY anomalies as a discovery queue to investigate emerging or unexpected patterns."
    )
    print(
        "Use RULE_ONLY anomalies as a known-risk queue to ensure business-defined risk patterns "
        "are still reviewed even when the model does not flag them."
    )

def add_review_priority(df):
    df = df.copy()

    def assign_priority(row):
        if row["anomaly_source"] == "RULE_AND_MODEL":
            return "P1_HIGH"
        if row["anomaly_source"] == "MODEL_ONLY":
            return "P2_INVESTIGATE"
        if row["anomaly_source"] == "RULE_ONLY":
            return "P3_KNOWN_RISK"
        return "P4_NORMAL"

    df["review_priority"] = df.apply(assign_priority, axis=1)

    return df

def print_review_workload_estimate(df):
    print("\nReview Workload Estimate")
    print("------------------------")

    review_df = df[df["review_priority"] != "P4_NORMAL"]

    print(f"Total payments: {len(df)}")
    print(f"Payments requiring anomaly review: {len(review_df)}")
    print(f"Review workload rate: {len(review_df) / len(df):.2%}")

    print("\nReview queue counts:")
    print(review_df["review_priority"].value_counts())

def print_review_priority_summary(df):
    print("\nReview Priority Summary")
    print("-----------------------")
    print(df["review_priority"].value_counts())

def main():
    df = load_scored_anomalies()
    df = add_review_priority(df)

    print_anomaly_source_summary(df)
    print_recommended_action_summary(df)
    print_review_priority_summary(df)
    print_combined_anomaly_summary(df)
    print_top_anomaly_reasons(df)
    print_amount_summary_by_anomaly_source(df)
    print_failure_rate_by_anomaly_source(df)
    print_high_priority_review_queue(df)
    print_model_only_review_queue(df)
    print_rule_only_review_queue(df)
    print_review_workload_estimate(df)
    print_business_interpretation(df)
    print_recommendation()


if __name__ == "__main__":
    main()
