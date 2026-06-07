import pandas as pd


payments = [
    {"payment_id": "P1001", "amount": 250000, "currency": "USD", "status": "SETTLED"},
    {"payment_id": "P1002", "amount": 980000, "currency": "EUR", "status": "FAILED"},
    {"payment_id": "P1003", "amount": 12500, "currency": "USD", "status": "PENDING"},
    {"payment_id": "P1004", "amount": 750000, "currency": "GBP", "status": "FAILED"},
    {"payment_id": "P1005", "amount": 43000, "currency": "USD", "status": "SETTLED"},
    {"payment_id": "P1006", "amount": 1200000, "currency": "EUR", "status": "SETTLED"},
]


df = pd.DataFrame(payments)

print("All payments:")
print(df)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataFrame info:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nSelected columns:")
print(df[["payment_id", "amount", "status"]])

print("\nFailed payments:")
failed_payments = df[df["status"] == "FAILED"]
print(failed_payments)

print("\nHigh value payments (amount > 100000):")
high_value_payments = df[df["amount"] > 100000]
print(high_value_payments)

print("\nFailed payments above 500000:")
failed_high_value_payments = df[(df["status"] == "FAILED") & (df["amount"] > 500000)]
print(failed_high_value_payments)

print("\nsorted by amount:")
sorted_payments = df.sort_values(by="amount", ascending=False)
print(sorted_payments)

print("\nTotal amount by currency:")
total_by_currency = df.groupby("currency")["amount"].sum()
print(total_by_currency)

print("\ncount of payments by status:")
count_by_status = df.groupby("status")["payment_id"].count()
print(count_by_status)

print("\nAverage amount by currency:")
average_by_currency = df.groupby("currency")["amount"].mean()
print(average_by_currency)

df["is_high_value"] = df["amount"] > 100000
df["amount_bucket"] = pd.cut(
    df["amount"],
    bins=[0, 100000, 500000, 2000000],
    labels=["LOW", "MEDIUM", "HIGH"]
)
print("\nDataFrame with high value flag:")
print(df)

df.to_csv("week-00/payments_sample.csv", index=False)

df_from_csv = pd.read_csv("week-00/payments_sample.csv")

print("\nData read back from CSV:")
print(df_from_csv)