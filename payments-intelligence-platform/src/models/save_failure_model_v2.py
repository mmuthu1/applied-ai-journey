from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/processed/payments_failure_features_v2.csv")
MODEL_PATH = Path("models/payment_failure_classifier_v2.pkl")

TARGET_COLUMN = "is_failed"

ID_COLUMNS = [
    "payment_id",
    "payment_date",
]

LEAKAGE_COLUMNS = [
    "status",
    "failure_reason",
]

RECOMMENDED_THRESHOLD = 0.55

MODEL_VERSION = "v2"

BASELINE_COMPARISON = {
    "week2_baseline": {
        "precision": 0.0779,
        "recall": 0.5357,
        "f1": 0.1361,
        "false_positives": 355,
        "false_negatives": 26,
        "alert_rate": 0.385,
    },
    "v2_threshold_0_55": {
        "precision": 0.1049,
        "recall": 0.5000,
        "f1": 0.1734,
        "false_positives": 239,
        "false_negatives": 28,
        "alert_rate": 0.267,
    },
}


def load_v2_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"V2 feature dataset not found: {DATA_PATH}. "
            "Run python -m src.models.build_failure_features_v2 first."
        )

    df = pd.read_csv(DATA_PATH)

    print("Loaded v2 payment failure feature dataset")
    print("-----------------------------------------")
    print(f"Path: {DATA_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Failure rate: {df[TARGET_COLUMN].mean():.2%}")

    return df


def prepare_features_and_target(df):
    columns_to_drop = [
        TARGET_COLUMN,
        *ID_COLUMNS,
    ]

    columns_to_drop = [
        column for column in columns_to_drop if column in df.columns
    ]

    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN]

    leakage_found = [
        column for column in LEAKAGE_COLUMNS if column in X.columns
    ]

    if leakage_found:
        raise ValueError(f"Leakage columns found in features: {leakage_found}")

    return X, y


def identify_feature_types(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    boolean_features = X.select_dtypes(include=["bool"]).columns.tolist()

    print("\nFeature groups")
    print("--------------")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Boolean features: {len(boolean_features)}")
    print(f"Total model features before encoding: {X.shape[1]}")

    return numeric_features, categorical_features, boolean_features


def build_model_pipeline(numeric_features, categorical_features, boolean_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("boolean", "passthrough", boolean_features),
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


def save_model_artifact(
    model_pipeline,
    feature_columns,
    numeric_features,
    categorical_features,
    boolean_features,
):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model_pipeline": model_pipeline,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "boolean_features": boolean_features,
        "target_column": TARGET_COLUMN,
        "model_version": MODEL_VERSION,
        "recommended_threshold": RECOMMENDED_THRESHOLD,
        "training_dataset": str(DATA_PATH),
        "model_type": "LogisticRegression",
        "class_weight": "balanced",
        "baseline_comparison": BASELINE_COMPARISON,
        "notes": (
            "V2 payment failure model trained on improved failure-risk features. "
            "Recommended threshold is 0.55 based on Week 6 threshold tuning."
        ),
    }

    joblib.dump(artifact, MODEL_PATH)

    print("\nSaved v2 payment failure model artifact")
    print("---------------------------------------")
    print(f"Path: {MODEL_PATH}")
    print(f"Model version: {MODEL_VERSION}")
    print(f"Recommended threshold: {RECOMMENDED_THRESHOLD}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Artifact keys: {list(artifact.keys())}")


def main():
    df = load_v2_dataset()
    X, y = prepare_features_and_target(df)

    numeric_features, categorical_features, boolean_features = identify_feature_types(X)

    model_pipeline = build_model_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        boolean_features=boolean_features,
    )

    print("\nTraining final v2 model on all available v2 data")
    print("------------------------------------------------")
    model_pipeline.fit(X, y)

    save_model_artifact(
        model_pipeline=model_pipeline,
        feature_columns=X.columns.tolist(),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        boolean_features=boolean_features,
    )


if __name__ == "__main__":
    main()
