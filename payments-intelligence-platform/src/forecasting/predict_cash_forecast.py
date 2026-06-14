from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/cash_forecast_model.pkl")


def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run python -m src.forecasting.save_cash_forecast_model first."
        )

    artifact = joblib.load(MODEL_PATH)

    print("Loaded cash forecast model artifact")
    print("-----------------------------------")
    print(f"Model path: {MODEL_PATH}")
    print(f"Target column: {artifact['target_column']}")
    print(f"Feature count: {len(artifact['feature_columns'])}")
    print(f"Artifact keys: {list(artifact.keys())}")

    return artifact


def create_sample_daily_cash_records():
    records = [
        {
            "forecast_date": "2024-06-28",
            "daily_payment_count": 30,
            "daily_total_amount": 3500000,
            "daily_average_amount": 116666.67,
            "daily_median_amount": 65000,
            "failed_payment_count": 2,
            "high_value_payment_count": 1,
            "unique_currency_count": 6,
            "unique_country_count": 7,
            "failed_payment_rate": 2 / 30,
            "high_value_payment_rate": 1 / 30,
            "day_of_week": 4,
            "month": 6,
            "day_of_month": 28,
            "is_weekend": False,
            "previous_day_total_amount": 3600000,
            "previous_day_payment_count": 29,
            "rolling_3_day_avg_amount": 3400000,
            "rolling_7_day_avg_amount": 3550000,
            "rolling_3_day_payment_count": 28,
            "rolling_7_day_payment_count": 30,
        },
        {
            "forecast_date": "2024-06-29",
            "daily_payment_count": 18,
            "daily_total_amount": 1800000,
            "daily_average_amount": 100000,
            "daily_median_amount": 58000,
            "failed_payment_count": 0,
            "high_value_payment_count": 0,
            "unique_currency_count": 4,
            "unique_country_count": 5,
            "failed_payment_rate": 0,
            "high_value_payment_rate": 0,
            "day_of_week": 5,
            "month": 6,
            "day_of_month": 29,
            "is_weekend": True,
            "previous_day_total_amount": 3500000,
            "previous_day_payment_count": 30,
            "rolling_3_day_avg_amount": 3100000,
            "rolling_7_day_avg_amount": 3300000,
            "rolling_3_day_payment_count": 26,
            "rolling_7_day_payment_count": 29,
        },
        {
            "forecast_date": "2024-06-30",
            "daily_payment_count": 42,
            "daily_total_amount": 6200000,
            "daily_average_amount": 147619.05,
            "daily_median_amount": 85000,
            "failed_payment_count": 4,
            "high_value_payment_count": 5,
            "unique_currency_count": 6,
            "unique_country_count": 7,
            "failed_payment_rate": 4 / 42,
            "high_value_payment_rate": 5 / 42,
            "day_of_week": 6,
            "month": 6,
            "day_of_month": 30,
            "is_weekend": True,
            "previous_day_total_amount": 1800000,
            "previous_day_payment_count": 18,
            "rolling_3_day_avg_amount": 3833333.33,
            "rolling_7_day_avg_amount": 3900000,
            "rolling_3_day_payment_count": 30,
            "rolling_7_day_payment_count": 31,
        },
    ]

    return pd.DataFrame(records)


def assign_forecast_band(predicted_amount):
    if predicted_amount >= 5_000_000:
        return "HIGH"
    if predicted_amount >= 3_000_000:
        return "MEDIUM"
    return "LOW"


def recommend_action(forecast_band):
    if forecast_band == "HIGH":
        return "Review liquidity, staffing, and high-value payment queue"
    if forecast_band == "MEDIUM":
        return "Monitor expected payment activity and exception volume"
    return "Normal cash operations planning"


def predict_next_day_cash_amounts(artifact, cash_activity_df):
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    feature_df = cash_activity_df[feature_columns]

    predictions = model.predict(feature_df)

    results_df = cash_activity_df.copy()
    results_df["predicted_next_day_total_amount"] = predictions
    results_df["forecast_band"] = results_df[
        "predicted_next_day_total_amount"
    ].apply(assign_forecast_band)
    results_df["recommended_action"] = results_df["forecast_band"].apply(
        recommend_action
    )
    results_df["forecast_vs_7_day_avg"] = (
        results_df["predicted_next_day_total_amount"]
        - results_df["rolling_7_day_avg_amount"]
    )

    results_df["forecast_vs_7_day_avg_pct"] = (
        results_df["forecast_vs_7_day_avg"]
        / results_df["rolling_7_day_avg_amount"]
    ) * 100

    return results_df


def print_predictions(results_df):
    display_columns = [
        "forecast_date",
        "daily_payment_count",
        "daily_total_amount",
        "rolling_7_day_avg_amount",
        "failed_payment_rate",
        "high_value_payment_rate",
        "predicted_next_day_total_amount",
        "forecast_band",
        "recommended_action",
        "forecast_vs_7_day_avg",
        "forecast_vs_7_day_avg_pct",
    ]

    display_df = results_df[display_columns].copy()

    money_columns = [
        "daily_total_amount",
        "rolling_7_day_avg_amount",
        "predicted_next_day_total_amount",
    ]

    for column in money_columns:
        display_df[column] = display_df[column].round(2)

    print("\nCash Forecast Predictions")
    print("-------------------------")
    print(display_df.to_string(index=False))

    print("\nBusiness interpretation")
    print("-----------------------")

    for index, row in results_df.iterrows():
        print(
            f"Record {index + 1}: forecast date {row['forecast_date']} "
            f"has a predicted next-day total amount of "
            f"${row['predicted_next_day_total_amount']:,.2f}. "
            f"Forecast band: {row['forecast_band']}. "
            f"Action: {row['recommended_action']}."
        )


def main():
    artifact = load_model_artifact()

    print("\nCreating sample daily cash activity records...")
    cash_activity_df = create_sample_daily_cash_records()

    print("Scoring sample daily cash activity records...")
    results_df = predict_next_day_cash_amounts(
        artifact=artifact,
        cash_activity_df=cash_activity_df,
    )

    print_predictions(results_df)

    print("\nForecast note")
    print("-------------")
    print(
        "This forecast model is experimental and was trained on synthetic data. "
        "It should be used for learning and portfolio demonstration, not production cash decisions."
    )


if __name__ == "__main__":
    main()
