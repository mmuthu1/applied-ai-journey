from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/processed/cash_forecast_daily.csv")


TARGET_COLUMN = "next_day_total_amount"


FEATURE_COLUMNS = [
    "daily_payment_count",
    "daily_total_amount",
    "daily_average_amount",
    "daily_median_amount",
    "failed_payment_count",
    "high_value_payment_count",
    "unique_currency_count",
    "unique_country_count",
    "failed_payment_rate",
    "high_value_payment_rate",
    "day_of_week",
    "month",
    "day_of_month",
    "is_weekend",
    "previous_day_total_amount",
    "previous_day_payment_count",
    "rolling_3_day_avg_amount",
    "rolling_7_day_avg_amount",
    "rolling_3_day_payment_count",
    "rolling_7_day_payment_count",
]


def load_dataset():
    df = pd.read_csv(INPUT_PATH)
    df["payment_day"] = pd.to_datetime(df["payment_day"])

    print("Loaded cash forecast dataset")
    print("----------------------------")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Date range: {df['payment_day'].min()} to {df['payment_day'].max()}")

    return df


def time_based_train_test_split(df, test_size=0.20):
    df = df.sort_values("payment_day").reset_index(drop=True)

    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    print("\nTime-based train/test split")
    print("---------------------------")
    print(f"Training rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Training date range: {train_df['payment_day'].min()} to {train_df['payment_day'].max()}")
    print(f"Test date range: {test_df['payment_day'].min()} to {test_df['payment_day'].max()}")

    return train_df, test_df


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_predictions(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


def build_linear_regression_model():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_random_forest_model():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
    )


def train_and_evaluate_models(train_df, test_df):
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    results = []

    # Baseline: 7-day rolling average
    baseline_predictions = test_df["rolling_7_day_avg_amount"]
    results.append(
        evaluate_predictions(
            "7-day moving average baseline",
            y_test,
            baseline_predictions,
        )
    )

    # Linear Regression
    linear_model = build_linear_regression_model()
    linear_model.fit(X_train, y_train)
    linear_predictions = linear_model.predict(X_test)

    results.append(
        evaluate_predictions(
            "Linear Regression",
            y_test,
            linear_predictions,
        )
    )

    # Random Forest Regressor
    random_forest_model = build_random_forest_model()
    random_forest_model.fit(X_train, y_train)
    random_forest_predictions = random_forest_model.predict(X_test)

    results.append(
        evaluate_predictions(
            "Random Forest Regressor",
            y_test,
            random_forest_predictions,
        )
    )

    prediction_df = test_df[
        [
            "payment_day",
            "daily_total_amount",
            TARGET_COLUMN,
            "rolling_7_day_avg_amount",
        ]
    ].copy()

    prediction_df["linear_regression_prediction"] = linear_predictions
    prediction_df["random_forest_prediction"] = random_forest_predictions
    prediction_df["baseline_error"] = (
        prediction_df[TARGET_COLUMN] - prediction_df["rolling_7_day_avg_amount"]
    )

    prediction_df["linear_regression_error"] = (
        prediction_df[TARGET_COLUMN] - prediction_df["linear_regression_prediction"]
    )

    prediction_df["random_forest_error"] = (
        prediction_df[TARGET_COLUMN] - prediction_df["random_forest_prediction"]
    )

    return pd.DataFrame(results), prediction_df, random_forest_model


def print_results(results_df):
    print("\nCash Forecast Model Evaluation")
    print("------------------------------")

    display_df = results_df.copy()
    display_df["mae"] = display_df["mae"].round(2)
    display_df["rmse"] = display_df["rmse"].round(2)
    display_df["mape"] = display_df["mape"].round(2)

    baseline_mae = results_df[
    results_df["model_name"] == "7-day moving average baseline"
    ]["mae"].iloc[0]

    display_df["mae_improvement_vs_baseline"] = (
        baseline_mae - results_df["mae"]
    ).round(2)

    display_df["mae_improvement_pct_vs_baseline"] = (
        ((baseline_mae - results_df["mae"]) / baseline_mae) * 100
    ).round(2)

    print(display_df.sort_values("mae").to_string(index=False))
    
    best_model = results_df.sort_values("mae").iloc[0]

    print("\nBest model by MAE:")
    print(
        f"{best_model['model_name']} "
        f"with MAE ${best_model['mae']:,.2f} "
        f"and MAPE {best_model['mape']:.2f}%"
    )


def print_sample_predictions(prediction_df):
    print("\nSample Forecast Predictions")
    print("---------------------------")

    display_df = prediction_df.copy()

    numeric_columns = [
        "daily_total_amount",
        TARGET_COLUMN,
        "rolling_7_day_avg_amount",
        "linear_regression_prediction",
        "random_forest_prediction",
        "baseline_error",
        "linear_regression_error",
        "random_forest_error",
    ]

    for column in numeric_columns:
        display_df[column] = display_df[column].round(2)

    print(display_df.head(10).to_string(index=False))


def print_feature_importance(random_forest_model):
    print("\nRandom Forest Feature Importance")
    print("--------------------------------")

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": random_forest_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print(importance_df.head(15).to_string(index=False))


def print_business_interpretation(results_df):
    baseline = results_df[
        results_df["model_name"] == "7-day moving average baseline"
    ].iloc[0]

    best = results_df.sort_values("mae").iloc[0]

    improvement = baseline["mae"] - best["mae"]
    improvement_percent = (improvement / baseline["mae"]) * 100

    print("\nBusiness Interpretation")
    print("-----------------------")

    if best["model_name"] == "7-day moving average baseline":
        print(
            "The ML models did not beat the 7-day moving average baseline. "
            "This means the simple baseline remains the best current forecasting method."
        )
    else:
        print(
            f"The best ML model improved MAE by ${improvement:,.2f} "
            f"({improvement_percent:.2f}%) compared with the 7-day baseline."
        )

    print(
        "Forecasting performance should be judged against the baseline, not in isolation."
    )

def print_recommendation(results_df):
    best = results_df.sort_values("mae").iloc[0]

    print("\nRecommendation")
    print("--------------")

    if best["model_name"] == "7-day moving average baseline":
        print(
            "Use the 7-day moving average as the current benchmark. "
            "Do not replace it with ML yet."
        )
        print(
            "Next improvement should focus on stronger features, such as weekday patterns, "
            "currency-specific aggregates, and high-value payment indicators."
        )
    else:
        print(
            f"Use {best['model_name']} as the current experimental forecasting model."
        )
        print(
            "Continue validating performance with more data and additional time windows."
        )

def main():
    df = load_dataset()

    train_df, test_df = time_based_train_test_split(df)

    results_df, prediction_df, random_forest_model = train_and_evaluate_models(
        train_df,
        test_df,
    )

    print_sample_predictions(prediction_df)
    print_results(results_df)
    print_feature_importance(random_forest_model)
    print_business_interpretation(results_df)
    print_recommendation(results_df)


if __name__ == "__main__":
    main()
