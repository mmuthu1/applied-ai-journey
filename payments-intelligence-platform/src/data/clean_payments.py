import pandas as pd


VALID_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD"}
VALID_PAYMENT_TYPES = {"ACH", "WIRE", "SWIFT", "RTP"}
VALID_CHANNELS = {"ONLINE", "BATCH", "API", "FILE", "MOBILE"}
VALID_SETTLEMENT_WINDOWS = {"SAME_DAY", "NEXT_DAY", "T_PLUS_2"}
VALID_STATUSES = {"SETTLED", "FAILED", "PENDING"}


def standardize_text_columns(df):
    text_columns = [
        "currency",
        "country",
        "payment_type",
        "channel",
        "settlement_window",
        "status",
        "failure_reason",
    ]

    for column in text_columns:
        df[column] = df[column].astype(str).str.strip().str.upper()

    return df


def clean_payments(input_path, output_path):
    df = pd.read_csv(input_path)

    raw_count = len(df)

    df = df.drop_duplicates(subset=["payment_id"])

    after_duplicate_count = len(df)
    duplicates_removed = raw_count - after_duplicate_count

    df = standardize_text_columns(df)

    df["payment_date"] = pd.to_datetime(df["payment_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["counterparty_risk_score"] = pd.to_numeric(
        df["counterparty_risk_score"],
        errors="coerce",
    )
    df["historical_failure_count"] = pd.to_numeric(
        df["historical_failure_count"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "payment_id",
            "payment_date",
            "amount",
            "currency",
            "country",
            "payment_type",
            "channel",
            "counterparty_risk_score",
            "historical_failure_count",
            "settlement_window",
            "status",
            "is_failed",
        ]
    )

    df = df[df["amount"] > 0]
    df = df[df["counterparty_risk_score"].between(0, 1)]
    df = df[df["historical_failure_count"] >= 0]

    df = df[df["currency"].isin(VALID_CURRENCIES)]
    df = df[df["payment_type"].isin(VALID_PAYMENT_TYPES)]
    df = df[df["channel"].isin(VALID_CHANNELS)]
    df = df[df["settlement_window"].isin(VALID_SETTLEMENT_WINDOWS)]
    df = df[df["status"].isin(VALID_STATUSES)]

    df["failure_reason"] = df["failure_reason"].fillna("NONE")

    non_failed_mask = df["is_failed"] == False
    df.loc[non_failed_mask, "failure_reason"] = "NONE"

    df.to_csv(output_path, index=False)

    clean_count = len(df)
    records_removed = raw_count - clean_count

    print("Payment Data Cleaning Report")
    print("----------------------------")
    print(f"Raw records: {raw_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Final clean records: {clean_count}")
    print(f"Total records removed: {records_removed}")
    print(f"Clean file saved to: {output_path}")
    print(f"Failure rate after cleaning: {df['is_failed'].mean():.2%}")
    print(f"Minimum amount: {df['amount'].min():,.2f}")
    print(f"Maximum amount: {df['amount'].max():,.2f}")
    print(f"Average amount: {df['amount'].mean():,.2f}")
    print(f"Median amount: {df['amount'].median():,.2f}")
    print("\nStatus distribution after cleaning:")
    print(df["status"].value_counts())

    return df


def main():
    input_path = "data/raw/payments_raw.csv"
    output_path = "data/processed/payments_clean.csv"

    clean_payments(input_path, output_path)


if __name__ == "__main__":
    main()
