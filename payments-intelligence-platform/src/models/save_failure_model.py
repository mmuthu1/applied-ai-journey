from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/processed/payments_features.csv")
MODEL_OUTPUT_PATH = Path("models/payment_failure_classifier.pkl")
TARGET_COLUMN = "is_failed"


NUMERIC_FEATURES = [
    "amount",
    "counterparty_risk_score",
    "historical_failure_count",
    "payment_year",
    "payment_month",
    "payment_day",
    "day_of_week",
    "hour_of_day",
    "amount_log",
    "settlement_speed_days",
    "risk_adjusted_amount",
]


CATEGORICAL_FEATURES = [
    "currency",
    "country",
    "payment_type",
    "channel",
    "settlement_window",
    "amount_bucket",
    "risk_band",
    "prior_failure_band",
]


BOOLEAN_FEATURES = [
    "is_weekend",
    "is_high_value",
    "has_prior_failures",
    "is_large_international",
]


FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    print("Loaded feature dataset")
    print("----------------------")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Failure rate: {df[TARGET_COLUMN].mean():.2%}")

    return df


def prepare_features_and_target(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y


def build_model_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model_pipeline = build_model_pipeline()

    print("\nTraining final model pipeline...")
    model_pipeline.fit(X_train, y_train)

    print("Model training completed")

    return model_pipeline


def save_model(model_pipeline):
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_pipeline, MODEL_OUTPUT_PATH)

    print("\nModel saved")
    print("-----------")
    print(f"Model path: {MODEL_OUTPUT_PATH}")


def verify_saved_model(X):
    print("\nVerifying saved model")
    print("---------------------")

    loaded_model = joblib.load(MODEL_OUTPUT_PATH)

    sample_records = X.head(5)

    predicted_classes = loaded_model.predict(sample_records)
    predicted_probabilities = loaded_model.predict_proba(sample_records)[:, 1]

    verification_df = sample_records.copy()
    verification_df["predicted_is_failed"] = predicted_classes
    verification_df["predicted_failure_probability"] = predicted_probabilities

    print(verification_df.to_string(index=False))


def main():
    df = load_dataset()

    X, y = prepare_features_and_target(df)

    model_pipeline = train_model(X, y)

    save_model(model_pipeline)

    verify_saved_model(X)

    print("\nModel artifact creation completed successfully")


if __name__ == "__main__":
    main()
