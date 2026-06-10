from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/cash_forecast_daily.csv")


def load_dataset():
    df = pd.read_csv(INPUT_PATH)
    df["payment_day"] = pd.to_datetime(df["payment_day"])

    print("Loaded cash forecast dataset")
    print("----------------------------")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df['payment_day'].min()} to {df['payment_day'].max()}")

    return df


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_forecast(df, forecast_column):
    y_true = df["next_day_total_amount"]
    y_pred = df[forecast_column]

    return {
        "forecast_method": forecast_column,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def create_baseline_forecasts(df):
    forecast_df = df.copy()

    forecast_df["forecast_previous_day_amount"] = forecast_df[
        "daily_total_amount"
    ]

    forecast_df["forecast_3_day_avg_amount"] = forecast_df[
        "rolling_3_day_avg_amount"
    ]

    forecast_df["forecast_7_day_avg_amount"] = forecast_df[
        "rolling_7_day_avg_amount"
    ]
    forecast_df["error_previous_day_amount"] = (
        forecast_df["next_day_total_amount"]
        - forecast_df["forecast_previous_day_amount"]
    )

    forecast_df["absolute_error_previous_day_amount"] = (
        forecast_df["error_previous_day_amount"].abs()
    )
    
    forecast_df["error_3_day_amount"] = (
        forecast_df["next_day_total_amount"]
        - forecast_df["forecast_3_day_avg_amount"]
    )
    forecast_df["absolute_error_3_day_amount"] = (
        forecast_df["error_3_day_amount"].abs()
    )
    forecast_df["error_7_day_amount"] = (
        forecast_df["next_day_total_amount"]
        - forecast_df["forecast_7_day_avg_amount"]
    )
    forecast_df["absolute_error_7_day_amount"] = (
        forecast_df["error_7_day_amount"].abs()
    )

    return forecast_df


def evaluate_all_baselines(forecast_df):
    forecast_columns = [
        "forecast_previous_day_amount",
        "forecast_3_day_avg_amount",
        "forecast_7_day_avg_amount",
    ]

    results = []

    for forecast_column in forecast_columns:
        results.append(evaluate_forecast(forecast_df, forecast_column))

    return pd.DataFrame(results)


def print_results(results_df):
    print("\nBaseline Forecast Evaluation")
    print("----------------------------")

    display_df = results_df.copy()
    display_df["mae"] = display_df["mae"].round(2)
    display_df["rmse"] = display_df["rmse"].round(2)
    display_df["mape"] = display_df["mape"].round(2)

    print(display_df.sort_values("mae").to_string(index=False))

    best_by_mae = results_df.sort_values("mae").iloc[0]
    best_by_mape = results_df.sort_values("mape").iloc[0]

    print("\nBest baseline by MAE:")
    print(
        f"{best_by_mae['forecast_method']} "
        f"with MAE ${best_by_mae['mae']:,.2f}"
    )

    print("\nBest baseline by MAPE:")
    print(
        f"{best_by_mape['forecast_method']} "
        f"with MAPE {best_by_mape['mape']:.2f}%"
    )


def print_sample_forecasts(forecast_df):
    print("\nSample Forecasts")
    print("----------------")

    display_columns = [
        "payment_day",
        "daily_total_amount",
        "next_day_total_amount",
        "forecast_previous_day_amount",
        "forecast_3_day_avg_amount",
        "forecast_7_day_avg_amount",
    ]

    print(forecast_df[display_columns].head(10).to_string(index=False))


def print_business_interpretation(results_df):
    best = results_df.sort_values("mae").iloc[0]

    print("\nBusiness Interpretation")
    print("-----------------------")
    print(
        "Baseline forecasts provide a simple comparison point before using ML. "
        "Any machine learning forecast should beat these baselines to be useful."
    )
    print(
        f"The current best baseline by MAE is {best['forecast_method']} "
        f"with an average absolute error of ${best['mae']:,.2f}."
    )
    print(
        "This means the forecast is off by that dollar amount on average."
    )

def print_error_direction_summary(forecast_df):
    print("\nForecast Error Direction Summary")
    print("--------------------------------")

    forecast_columns = [
        "forecast_previous_day_amount",
        "forecast_3_day_avg_amount",
        "forecast_7_day_avg_amount",
    ]

    for column in forecast_columns:
        error = forecast_df["next_day_total_amount"] - forecast_df[column]

        under_forecast_count = (error > 0).sum()
        over_forecast_count = (error < 0).sum()

        print(f"\n{column}")
        print(f"Under-forecast days: {under_forecast_count}")
        print(f"Over-forecast days: {over_forecast_count}")

def print_recommendation(results_df):
    best = results_df.sort_values("mae").iloc[0]

    print("\nRecommendation")
    print("--------------")
    print(
        f"Use {best['forecast_method']} as the baseline to beat in the ML forecasting model."
    )
    print(
        "The next ML model should be compared against this baseline using MAE, RMSE, and MAPE."
    )
    
def main():
    df = load_dataset()

    forecast_df = create_baseline_forecasts(df)

    results_df = evaluate_all_baselines(forecast_df)

    print_sample_forecasts(forecast_df)
    print_results(results_df)
    print_business_interpretation(results_df)
    print_error_direction_summary(forecast_df)
    print_recommendation(results_df)
if __name__ == "__main__":
    main()
