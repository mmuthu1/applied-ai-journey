import pandas as pd


payments = [
    {
        "payment_id": "P1001",
        "amount": 250000,
        "currency": "USD",
        "country": "US",
        "payment_type": "WIRE",
        "channel": "API",
        "status": "SETTLED",
    },
    {
        "payment_id": "P1002",
        "amount": 980000,
        "currency": "EUR",
        "country": "DE",
        "payment_type": "SWIFT",
        "channel": "FILE",
        "status": "FAILED",
    },
    {
        "payment_id": "P1003",
        "amount": 12500,
        "currency": "USD",
        "country": "US",
        "payment_type": "ACH",
        "channel": "BATCH",
        "status": "PENDING",
    },
    {
        "payment_id": "P1004",
        "amount": 750000,
        "currency": "GBP",
        "country": "GB",
        "payment_type": "SWIFT",
        "channel": "FILE",
        "status": "FAILED",
    },
    {
        "payment_id": "P1005",
        "amount": 43000,
        "currency": "USD",
        "country": "US",
        "payment_type": "ACH",
        "channel": "ONLINE",
        "status": "SETTLED",
    },
    {
        "payment_id": "P1006",
        "amount": 1200000,
        "currency": "EUR",
        "country": "FR",
        "payment_type": "WIRE",
        "channel": "API",
        "status": "SETTLED",
    },
]


df = pd.DataFrame(payments)

print("All payments:")
print(df)

print("\nFailed payments:")
print(df[df["status"] == "FAILED"])

print("\nHigh-value payments:")
print(df[df["amount"] > 100000])

print("\nTotal amount by currency:")
print(df.groupby("currency")["amount"].sum())

print("\nPayment count by status:")
print(df.groupby("status")["payment_id"].count())

print("\nFailure count by payment type:")
failed_df = df[df["status"] == "FAILED"]
print(failed_df.groupby("payment_type")["payment_id"].count())

print("\nAverage amount by country:")
print(df.groupby("country")["amount"].mean())

print("\n Payment count by channel:")
print(df.groupby("channel")["payment_id"].count())

print("\n Total amount by payment type:")
print(df.groupby("payment_type")["amount"].sum())

print("\n Failed high-value payments:")
failed_high_value_df = df[(df["status"] == "FAILED") & (df["amount"] > 500000)]
print(failed_high_value_df)
