import pandas as pd


REQUIRED_COLUMNS = [
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
    "failure_reason",
    "is_failed",
]


VALID_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD"}
VALID_PAYMENT_TYPES = {"ACH", "WIRE", "SWIFT", "RTP"}
VALID_CHANNELS = {"ONLINE", "BATCH", "API", "FILE", "MOBILE"}
VALID_SETTLEMENT_WINDOWS = {"SAME_DAY", "NEXT_DAY", "T_PLUS_2"}
VALID_STATUSES = {"SETTLED", "FAILED", "PENDING"}


def validate_required_columns(df):
    missing_columns = []

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            missing_columns.append(column)

    return missing_columns


def validate_missing_values(df):
    return df.isnull().sum()


def validate_duplicate_payment_ids(df):
    return df["payment_id"].duplicated().sum()


def validate_positive_amounts(df):
    invalid_amounts = df[df["amount"] <= 0]
    return len(invalid_amounts)


def validate_categorical_values(df, column_name, allowed_values):
    invalid_rows = df[~df[column_name].isin(allowed_values)]
    return len(invalid_rows)


def validate_risk_score_range(df):
    invalid_rows = df[
        (df["counterparty_risk_score"] < 0)
        | (df["counterparty_risk_score"] > 1)
    ]

    return len(invalid_rows)


def validate_historical_failure_count(df):
    invalid_rows = df[df["historical_failure_count"] < 0]
    return len(invalid_rows)


def validate_payment_dates(df):
    parsed_dates = pd.to_datetime(df["payment_date"], errors="coerce")
    invalid_dates = parsed_dates.isnull().sum()
    return invalid_dates

def validate_is_failed_reason_consistency(df):
    inconsistent_rows = df[
        ((df["is_failed"] == True) & (df["failure_reason"] == "NONE"))
        | ((df["is_failed"] == False) & (df["failure_reason"] != "NONE"))
    ]

    return len(inconsistent_rows)

def run_validation(input_path):
    df = pd.read_csv(input_path)

    print("Payment Data Validation Report")
    print("------------------------------")
    print(f"Total records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\n1. Required columns check")
    missing_columns = validate_required_columns(df)
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
    else:
        print("All required columns are present")

    print("\n2. Missing values by column")
    print(validate_missing_values(df))

    print("\n3. Duplicate payment_id count")
    print(validate_duplicate_payment_ids(df))

    print("\n4. Invalid amount count")
    print(validate_positive_amounts(df))

    print("\n5. Invalid currency count")
    print(validate_categorical_values(df, "currency", VALID_CURRENCIES))

    print("\n6. Invalid payment_type count")
    print(validate_categorical_values(df, "payment_type", VALID_PAYMENT_TYPES))

    print("\n7. Invalid channel count")
    print(validate_categorical_values(df, "channel", VALID_CHANNELS))

    print("\n8. Invalid settlement_window count")
    print(validate_categorical_values(df, "settlement_window", VALID_SETTLEMENT_WINDOWS))

    print("\n9. Invalid status count")
    print(validate_categorical_values(df, "status", VALID_STATUSES))

    print("\n10. Invalid counterparty_risk_score count")
    print(validate_risk_score_range(df))

    print("\n11. Invalid historical_failure_count count")
    print(validate_historical_failure_count(df))

    print("\n12. Invalid payment_date count")
    print(validate_payment_dates(df))

    print("\n13. Inconsistent is_failed and failure_reason values")
    print(validate_is_failed_reason_consistency(df))


def main():
    print("VALIDATING RAW DATA")
    run_validation("data/raw/payments_raw.csv")

    print("\n\nVALIDATING CLEAN DATA")
    run_validation("data/processed/payments_clean.csv")


if __name__ == "__main__":
    main()
