from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


INPUT_PATH = Path("data/processed/cash_forecast_daily.csv")
MODEL_OUTPUT_PATH = Path("models/cash_forecast_model.pkl")

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


def prepare_features_and_target(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y


def build_forecast_model():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
    )


def train_final_model(X, y):
    model = build_forecast_model()

    print("\nTraining final Random Forest forecast model...")
    model.fit(X, y)
    print("Model training completed")

    return model


def save_model(model):
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        },
        MODEL_OUTPUT_PATH,
    )

    print("\nModel saved")
    print("-----------")
    print(f"Model path: {MODEL_OUTPUT_PATH}")


def verify_saved_model(X):
    print("\nVerifying saved model")
    print("---------------------")

    saved_artifact = joblib.load(MODEL_OUTPUT_PATH)

    model = saved_artifact["model"]
    feature_columns = saved_artifact["feature_columns"]

    sample_records = X[feature_columns].head(5)
    predictions = model.predict(sample_records)

    verification_df = sample_records.copy()
    verification_df["predicted_next_day_total_amount"] = predictions

    print(verification_df.to_string(index=False))
    print("\nSaved model verification completed successfully")


def main():
    df = load_dataset()

    X, y = prepare_features_and_target(df)

    model = train_final_model(X, y)

    save_model(model)

    verify_saved_model(X)

    print("\nCash forecast model artifact creation completed successfully")


if __name__ == "__main__":
    main()
