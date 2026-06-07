payments = [
    {"payment_id": "P1001", "amount": 250000, "currency": "USD", "status": "SETTLED"},
    {"payment_id": "P1002", "amount": 980000, "currency": "EUR", "status": "FAILED"},
    {"payment_id": "P1003", "amount": 12500, "currency": "USD", "status": "PENDING"},
    {"payment_id": "P1004", "amount": 750000, "currency": "GBP", "status": "FAILED"},
    {"payment_id": "P1005", "amount": 43000, "currency": "USD", "status": "SETTLED"},
    {"payment_id": "P1006", "amount": 1200000, "currency": "EUR", "status": "SETTLED"},
]

print("All payment IDs:")
for payment in payments:
    print(payment["payment_id"])

def get_failed_payments(payments):
    return [payment for payment in payments if payment["status"] == "FAILED"]
failed_payments = get_failed_payments(payments)
print("Failed Payments:")
for payment in failed_payments:
    print(payment["payment_id"], payment["amount"], payment["currency"])

def get_high_value_payments(payments, threshold):
    return [payment for payment in payments if payment["amount"] > threshold]
high_value_payments = get_high_value_payments(payments, 100000)
print("High Value Payments:")
for payment in high_value_payments:
    print(payment["payment_id"], payment["amount"], payment["currency"])

def get_total_amount_by_currency(payments, currency):
    total = sum(payment["amount"] for payment in payments if payment["currency"] == currency)
    return total
total_usd = get_total_amount_by_currency(payments, "USD")
print(f"Total amount in USD: {total_usd}")
total_eur = get_total_amount_by_currency(payments, "EUR")
print(f"Total amount in EUR: {total_eur}")
total_gbp = get_total_amount_by_currency(payments, "GBP")
print(f"Total amount in GBP: {total_gbp}")

def count_payments_by_status(payments):
    status_counts = {}
    for payment in payments:
        status = payment["status"]
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts[status] = 1
    return status_counts
payment_status_counts = count_payments_by_status(payments)
print("Payment Status Counts:")
for status, count in payment_status_counts.items():
    print(f"{status}: {count}")

def get_average_payment_amount(payments):
    total_amount = sum(payment["amount"] for payment in payments)
    average = total_amount / len(payments) if payments else 0
    return average
average_amount = get_average_payment_amount(payments)
print("Average payment amount: ", average_amount)

def get_payments_by_currency(payments, currency):
    return [payment for payment in payments if payment["currency"] == currency]
usd_payments = get_payments_by_currency(payments, "USD")
print("USD Payments:")
for payment in usd_payments:
    print(payment["payment_id"], payment["amount"], payment["status"])
