from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_PATH = Path("data/processed/payment_anomaly_features.csv")
MODEL_OUTPUT_PATH = Path("models/payment_anomaly_detector.pkl")


NUMERIC_FEATURES = [
    "amount",
    "counterparty_risk_score",
    "historical_failure_count",
    "risk_adjusted_amount",
    "amount_log",
    "settlement_speed_days",
    "amount_z_score",
    "risk_adjusted_amount_z_score",
    "amount_percentile_rank",
    "risk_score_percentile_rank",
    "country_frequency",
    "currency_frequency",
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
    "is_extreme_amount",
    "is_top_1pct_amount",
    "is_extreme_risk_adjusted_amount",
    "is_top_1pct_risk_score",
    "is_high_risk_large_payment",
    "is_large_file_payment",
    "is_high_risk_swift_payment",
    "is_prior_failure_high_value",
    "is_rare_country",
    "is_rare_currency",
]


FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES


def load_dataset():
    df = pd.read_csv(INPUT_PATH)

    print("Loaded anomaly feature dataset")
    print("------------------------------")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Rule-based anomaly rate: {df['is_rule_based_anomaly'].mean():.2%}")

    return df


def build_anomaly_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
        ]
    )

    model = IsolationForest(
        n_estimators=300,
        contamination=0.04,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_model(df):
    X = df[FEATURE_COLUMNS]

    model_pipeline = build_anomaly_pipeline()

    print("\nTraining final Isolation Forest anomaly model...")
    model_pipeline.fit(X)
    print("Model training completed")

    return model_pipeline, X


def save_model(model_pipeline):
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_pipeline": model_pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "boolean_features": BOOLEAN_FEATURES,
        "contamination": 0.04,
        "model_type": "IsolationForest",
    }

    joblib.dump(artifact, MODEL_OUTPUT_PATH)

    print("\nModel artifact saved")
    print("--------------------")
    print(f"Model path: {MODEL_OUTPUT_PATH}")


def verify_saved_model(X):
    print("\nVerifying saved model artifact")
    print("------------------------------")

    artifact = joblib.load(MODEL_OUTPUT_PATH)

    model_pipeline = artifact["model_pipeline"]
    feature_columns = artifact["feature_columns"]

    sample_records = X[feature_columns].head(5)

    raw_predictions = model_pipeline.predict(sample_records)
    anomaly_scores = model_pipeline.decision_function(sample_records)

    verification_df = sample_records.copy()
    verification_df["raw_isolation_forest_prediction"] = raw_predictions
    verification_df["is_anomaly"] = raw_predictions == -1
    verification_df["anomaly_score"] = anomaly_scores

    print(f"Artifact keys: {list(artifact.keys())}")
    print("\nSample verification predictions:")
    print(verification_df.to_string(index=False))

    print("\nSaved anomaly model verification completed successfully")


def main():
    df = load_dataset()

    model_pipeline, X = train_model(df)

    save_model(model_pipeline)

    verify_saved_model(X)

    print("\nPayment anomaly detector artifact creation completed successfully")


if __name__ == "__main__":
    main()
