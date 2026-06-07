import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


np.random.seed(42)

NUM_RECORDS = 5000

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]
COUNTRIES = ["US", "DE", "GB", "JP", "CA", "IN", "AU"]
PAYMENT_TYPES = ["ACH", "WIRE", "SWIFT", "RTP"]
CHANNELS = ["ONLINE", "BATCH", "API", "FILE", "MOBILE"]
SETTLEMENT_WINDOWS = ["SAME_DAY", "NEXT_DAY", "T_PLUS_2"]

FAILURE_REASONS = [
    "INSUFFICIENT_FUNDS",
    "INVALID_ACCOUNT",
    "COMPLIANCE_REVIEW",
    "NETWORK_ERROR",
    "SFTP_FILE_DELAY",
    "NONE",
]


def generate_payment_id():
    return "P-" + str(uuid.uuid4())[:8].upper()


def generate_payment_date():
    start_date = datetime(2024, 1, 1)

    random_days = np.random.randint(0, 180)
    random_hours = np.random.randint(0, 24)
    random_minutes = np.random.randint(0, 60)

    return start_date + timedelta(
        days=int(random_days),
        hours=int(random_hours),
        minutes=int(random_minutes),
    )


def calculate_failure_probability(
    amount,
    payment_type,
    channel,
    counterparty_risk_score,
    historical_failure_count,
):
    probability = 0.04

    if amount > 500000:
        probability += 0.03

    if amount > 1000000:
        probability += 0.04

    if payment_type == "SWIFT":
        probability += 0.03

    if channel == "FILE":
        probability += 0.02

    if counterparty_risk_score > 0.7:
        probability += 0.05

    if historical_failure_count >= 3:
        probability += 0.04

    return min(probability, 0.45)


def generate_payment_record():
    amount = round(float(np.random.lognormal(mean=11, sigma=1.2)), 2)
    currency = np.random.choice(CURRENCIES)
    country = np.random.choice(COUNTRIES)

    payment_type = np.random.choice(
        PAYMENT_TYPES,
        p=[0.35, 0.30, 0.25, 0.10],
    )

    channel = np.random.choice(
        CHANNELS,
        p=[0.22, 0.28, 0.25, 0.18, 0.07],
    )

    settlement_window = np.random.choice(
        SETTLEMENT_WINDOWS,
        p=[0.45, 0.40, 0.15],
    )

    counterparty_risk_score = round(float(np.random.beta(2, 5)), 3)
    historical_failure_count = int(np.random.poisson(1))

    failure_probability = calculate_failure_probability(
        amount=amount,
        payment_type=payment_type,
        channel=channel,
        counterparty_risk_score=counterparty_risk_score,
        historical_failure_count=historical_failure_count,
    )

    is_failed = bool(np.random.random() < failure_probability)

    if is_failed:
        status = "FAILED"
        failure_reason = np.random.choice(FAILURE_REASONS[:-1])
    else:
        status = np.random.choice(["SETTLED", "PENDING"], p=[0.93, 0.07])
        failure_reason = "NONE"

    return {
        "payment_id": generate_payment_id(),
        "payment_date": generate_payment_date(),
        "amount": amount,
        "currency": currency,
        "country": country,
        "payment_type": payment_type,
        "channel": channel,
        "counterparty_risk_score": counterparty_risk_score,
        "historical_failure_count": historical_failure_count,
        "settlement_window": settlement_window,
        "status": status,
        "failure_reason": failure_reason,
        "is_failed": is_failed,
    }


def generate_payments(num_records):
    records = []

    for _ in range(num_records):
        records.append(generate_payment_record())

    return pd.DataFrame(records)


def print_dataset_summary(df):
    print("Dataset summary")
    print("---------------")
    print(f"Total records: {len(df)}")
    print(f"Failure rate: {df['is_failed'].mean():.2%}")

    print("\nStatus counts:")
    print(df["status"].value_counts())

    print("\nPayment count by payment type:")
    print(df.groupby("payment_type")["payment_id"].count())

    print("\nFailure rate by payment type:")
    print(df.groupby("payment_type")["is_failed"].mean().sort_values(ascending=False))

    print("\nTotal amount by payment type:")
    print(df.groupby("payment_type")["amount"].sum().sort_values(ascending=False))

    print("\nPayment count by currency:")
    print(df["currency"].value_counts())

    print("\nFailure rate by channel:")
    print(df.groupby("channel")["is_failed"].mean().sort_values(ascending=False))

    print("\nHigh-value payment count above 500,000:")
    print((df["amount"] > 500000).sum())

    print("\nSample records:")
    print(df.head())


def main():
    df = generate_payments(NUM_RECORDS)

    output_path = "data/raw/payments_raw.csv"
    df.to_csv(output_path, index=False)

    print(f"Generated payment dataset and saved to: {output_path}")
    print()
    print_dataset_summary(df)


if __name__ == "__main__":
    main()