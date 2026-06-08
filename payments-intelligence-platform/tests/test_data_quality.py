from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/payments_raw.csv")
CLEAN_DATA_PATH = Path("data/processed/payments_clean.csv")
FEATURES_DATA_PATH = Path("data/processed/payments_features.csv")


VALID_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD"}
VALID_PAYMENT_TYPES = {"ACH", "WIRE", "SWIFT", "RTP"}
VALID_CHANNELS = {"ONLINE", "BATCH", "API", "FILE", "MOBILE"}


def test_raw_data_file_exists():
    assert RAW_DATA_PATH.exists()


def test_clean_data_file_exists():
    assert CLEAN_DATA_PATH.exists()


def test_features_data_file_exists():
    assert FEATURES_DATA_PATH.exists()


def test_features_dataset_has_records():
    df = pd.read_csv(FEATURES_DATA_PATH)
    assert len(df) > 0


def test_features_dataset_has_expected_record_count():
    df = pd.read_csv(FEATURES_DATA_PATH)
    assert len(df) == 5000


def test_payment_id_has_no_missing_values():
    df = pd.read_csv(FEATURES_DATA_PATH)
    assert df["payment_id"].isnull().sum() == 0


def test_no_duplicate_payment_ids():
    df = pd.read_csv(FEATURES_DATA_PATH)
    assert df["payment_id"].duplicated().sum() == 0


def test_amount_is_positive():
    df = pd.read_csv(FEATURES_DATA_PATH)
    assert (df["amount"] > 0).all()


def test_is_failed_column_exists():
    df = pd.read_csv(FEATURES_DATA_PATH)
    assert "is_failed" in df.columns


def test_is_failed_values_are_valid():
    df = pd.read_csv(FEATURES_DATA_PATH)

    valid_values = {True, False, 0, 1, "True", "False"}

    assert set(df["is_failed"].unique()).issubset(valid_values)


def test_failure_rate_is_reasonable():
    df = pd.read_csv(FEATURES_DATA_PATH)

    failure_rate = df["is_failed"].mean()

    assert 0.03 <= failure_rate <= 0.15


def test_currency_values_are_valid():
    df = pd.read_csv(FEATURES_DATA_PATH)

    assert set(df["currency"].unique()).issubset(VALID_CURRENCIES)


def test_payment_type_values_are_valid():
    df = pd.read_csv(FEATURES_DATA_PATH)

    assert set(df["payment_type"].unique()).issubset(VALID_PAYMENT_TYPES)


def test_channel_values_are_valid():
    df = pd.read_csv(FEATURES_DATA_PATH)

    assert set(df["channel"].unique()).issubset(VALID_CHANNELS)


def test_risk_score_is_between_zero_and_one():
    df = pd.read_csv(FEATURES_DATA_PATH)

    assert df["counterparty_risk_score"].between(0, 1).all()


def test_leakage_columns_are_removed():
    df = pd.read_csv(FEATURES_DATA_PATH)

    leakage_columns = {"status", "failure_reason"}

    assert leakage_columns.isdisjoint(set(df.columns))


def test_required_engineered_features_exist():
    df = pd.read_csv(FEATURES_DATA_PATH)

    required_features = {
        "amount_log",
        "is_high_value",
        "amount_bucket",
        "risk_band",
        "has_prior_failures",
        "prior_failure_band",
        "settlement_speed_days",
        "is_large_international",
        "risk_adjusted_amount",
    }

    assert required_features.issubset(set(df.columns))