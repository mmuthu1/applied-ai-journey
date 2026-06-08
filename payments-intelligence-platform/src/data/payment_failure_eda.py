import pandas as pd


INPUT_PATH = "data/processed/payments_features.csv"


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def calculate_failure_rate(df):
    return df["is_failed"].mean()


def failure_rate_by_group(df, group_column):
    summary = (
        df.groupby(group_column)
        .agg(
            payment_count=("payment_id", "count"),
            failed_count=("is_failed", "sum"),
            failure_rate=("is_failed", "mean"),
            average_amount=("amount", "mean"),
            total_amount=("amount", "sum"),
        )
        .sort_values("failure_rate", ascending=False)
    )

    return summary


def analyze_overall_failure_rate(df):
    print_section("1. Overall Failure Rate")

    total_records = len(df)
    failed_records = df["is_failed"].sum()
    failure_rate = calculate_failure_rate(df)

    print(f"Total payments: {total_records}")
    print(f"Failed payments: {failed_records}")
    print(f"Failure rate: {failure_rate:.2%}")


def analyze_group_failure_rates(df):
    print_section("2. Failure Rate by Payment Type")
    print(failure_rate_by_group(df, "payment_type"))

    print_section("3. Failure Rate by Channel")
    print(failure_rate_by_group(df, "channel"))

    print_section("4. Failure Rate by Currency")
    print(failure_rate_by_group(df, "currency"))

    print_section("5. Failure Rate by Country")
    print(failure_rate_by_group(df, "country"))

    print_section("6. Failure Rate by Amount Bucket")
    print(failure_rate_by_group(df, "amount_bucket"))

    print_section("7. Failure Rate by Risk Band")
    print(failure_rate_by_group(df, "risk_band"))

    print_section("8. Failure Rate by Prior Failure Band")
    print(failure_rate_by_group(df, "prior_failure_band"))

    print_section("9. Failure Rate by Settlement Window")
    print(failure_rate_by_group(df, "settlement_window"))


def analyze_business_rule_features(df):
    print_section("10. Failure Rate by High-Value Flag")
    print(failure_rate_by_group(df, "is_high_value"))

    print_section("11. Failure Rate by Large International Flag")
    print(failure_rate_by_group(df, "is_large_international"))

    print_section("12. Risk-Adjusted Amount by Failure Status")
    summary = (
        df.groupby("is_failed")
        .agg(
            payment_count=("payment_id", "count"),
            average_amount=("amount", "mean"),
            median_amount=("amount", "median"),
            average_risk_score=("counterparty_risk_score", "mean"),
            average_risk_adjusted_amount=("risk_adjusted_amount", "mean"),
        )
    )

    print(summary)
    print_section("13. Failure Rate by Weekend Flag")
    print(failure_rate_by_group(df, "is_weekend"))


def analyze_numeric_features(df):
    print_section("14. Numeric Feature Summary by Failure Status")

    numeric_columns = [
        "amount",
        "amount_log",
        "counterparty_risk_score",
        "historical_failure_count",
        "risk_adjusted_amount",
        "settlement_speed_days",
        "hour_of_day",
        "day_of_week",
    ]

    summary = df.groupby("is_failed")[numeric_columns].mean()
    print(summary)


def analyze_top_risk_adjusted_payments(df):
    print_section("15. Top 10 Payments by Risk-Adjusted Amount")

    columns = [
            "payment_id",
            "amount",
            "currency",
            "country",
            "payment_type",
            "channel",
            "counterparty_risk_score",
            "risk_adjusted_amount",
            "is_failed",
        ]
    top_risk_payments = df.sort_values("risk_adjusted_amount", ascending=False)[columns].head(10)
    print(top_risk_payments)

def print_insights():
    print_section("16. Initial ML / Business Insights")

    insights = [
        "1. The target variable is imbalanced because failed payments are a minority class.",
        "2. Accuracy alone will not be enough for Week 2 model evaluation; precision, recall, and F1 score will matter.",
        "3. Payment type and channel appear useful because the generator intentionally assigns higher risk to SWIFT and FILE payments.",
        "4. Risk-related features such as counterparty_risk_score, risk_band, historical_failure_count, and risk_adjusted_amount should be useful predictors.",
        "5. Amount-related features such as amount_log, amount_bucket, is_high_value, and is_large_international may help identify operationally risky payments.",
    ]

    for insight in insights:
        print(insight)

def print_model_readiness_notes():
    print_section("17. Week 2 Model Readiness Notes")

    notes = [
        "Target column for Week 2: is_failed",
        "Leakage columns status and failure_reason were removed from payments_features.csv.",
        "Categorical features will need encoding before model training.",
        "Numerical features may need scaling for some models.",
        "Because failed payments are a minority class, use stratified train/test split.",
    ]

    for note in notes:
        print("- " + note)

def main():
    df = pd.read_csv(INPUT_PATH)

    analyze_overall_failure_rate(df)
    analyze_group_failure_rates(df)
    analyze_business_rule_features(df)
    analyze_numeric_features(df)
    analyze_top_risk_adjusted_payments(df)
    print_insights()
    print_model_readiness_notes()


if __name__ == "__main__":
    main()
