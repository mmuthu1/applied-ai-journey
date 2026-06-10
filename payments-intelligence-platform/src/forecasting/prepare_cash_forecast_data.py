from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/payments_clean.csv")
OUTPUT_PATH = Path("data/processed/cash_forecast_daily.csv")


def load_payments():
    df = pd.read_csv(INPUT_PATH)
    df["payment_date"] = pd.to_datetime(df["payment_date"])
    df["payment_day"] = df["payment_date"].dt.date

    return df


def create_daily_aggregates(df):
    daily_df = (
        df.groupby("payment_day")
        .agg(
            daily_payment_count=("payment_id", "count"),
            daily_total_amount=("amount", "sum"),
            daily_average_amount=("amount", "mean"),
            daily_median_amount=("amount", "median"),
            failed_payment_count=("is_failed", "sum"),
            high_value_payment_count=("amount", lambda x: (x > 500000).sum()),
            unique_currency_count=("currency", "nunique"),
            unique_country_count=("country", "nunique"),
        )
        .reset_index()
    )

    daily_df["payment_day"] = pd.to_datetime(daily_df["payment_day"])
    daily_df["failed_payment_rate"] = (
        daily_df["failed_payment_count"] / daily_df["daily_payment_count"]
    )
    daily_df["high_value_payment_rate"] = (
        daily_df["high_value_payment_count"] / daily_df["daily_payment_count"]
    )

    return daily_df


def add_time_features(daily_df):
    daily_df["day_of_week"] = daily_df["payment_day"].dt.dayofweek
    daily_df["month"] = daily_df["payment_day"].dt.month
    daily_df["day_of_month"] = daily_df["payment_day"].dt.day
    daily_df["is_weekend"] = daily_df["day_of_week"].isin([5, 6])

    return daily_df


def add_lag_and_rolling_features(daily_df):
    daily_df = daily_df.sort_values("payment_day").reset_index(drop=True)

    daily_df["previous_day_total_amount"] = daily_df["daily_total_amount"].shift(1)
    daily_df["previous_day_payment_count"] = daily_df["daily_payment_count"].shift(1)

    daily_df["rolling_3_day_avg_amount"] = (
        daily_df["daily_total_amount"].rolling(window=3).mean()
    )

    daily_df["rolling_7_day_avg_amount"] = (
        daily_df["daily_total_amount"].rolling(window=7).mean()
    )

    daily_df["rolling_3_day_payment_count"] = (
        daily_df["daily_payment_count"].rolling(window=3).mean()
    )

    daily_df["rolling_7_day_payment_count"] = (
        daily_df["daily_payment_count"].rolling(window=7).mean()
    )

    return daily_df


def add_forecast_target(daily_df):
    daily_df["next_day_total_amount"] = daily_df["daily_total_amount"].shift(-1)

    return daily_df


def clean_forecast_dataset(daily_df):
    before_count = len(daily_df)

    daily_df = daily_df.dropna().reset_index(drop=True)

    after_count = len(daily_df)

    print("Forecast dataset cleaning")
    print("-------------------------")
    print(f"Rows before dropping NA: {before_count}")
    print(f"Rows after dropping NA: {after_count}")
    print(f"Rows removed: {before_count - after_count}")

    return daily_df


def print_forecast_dataset_summary(daily_df):
    print("\nCash Forecast Dataset Summary")
    print("-----------------------------")
    print(f"Rows: {len(daily_df)}")
    print(f"Columns: {len(daily_df.columns)}")
    print(f"Date range: {daily_df['payment_day'].min()} to {daily_df['payment_day'].max()}")

    print("\nColumns:")
    print(list(daily_df.columns))

    print("\nSample records:")
    print(daily_df.head().to_string(index=False))

    print("\nTarget summary: next_day_total_amount")
    print(daily_df["next_day_total_amount"].describe())


def create_cash_forecast_dataset():
    df = load_payments()

    daily_df = create_daily_aggregates(df)
    daily_df = add_time_features(daily_df)
    daily_df = add_lag_and_rolling_features(daily_df)
    daily_df = add_forecast_target(daily_df)
    daily_df = clean_forecast_dataset(daily_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved cash forecast dataset to: {OUTPUT_PATH}")
    print_forecast_dataset_summary(daily_df)

    return daily_df


def main():
    create_cash_forecast_dataset()


if __name__ == "__main__":
    main()
