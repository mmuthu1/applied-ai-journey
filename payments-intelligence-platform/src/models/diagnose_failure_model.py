from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/processed/payments_features.csv")

TARGET_COLUMN = "is_failed"

LEAKAGE_COLUMNS = [
    "status",
    "failure_reason",
]

ID_COLUMNS = [
    "payment_id",
    "payment_date",
]


def load_feature_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}. "
            "Run python -m src.data.run_pipeline first."
        )

    df = pd.read_csv(DATA_PATH)

    print("Loaded feature dataset")
    print("----------------------")
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

    for leakage_column in LEAKAGE_COLUMNS:
        if leakage_column in X.columns:
            raise ValueError(
                f"Leakage column found in features: {leakage_column}"
            )

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

    return numeric_features, categorical_features, boolean_features


def build_baseline_model(numeric_features, categorical_features, boolean_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
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


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = None

    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    print("\nCurrent Week 2-style baseline performance")
    print("-----------------------------------------")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 score:  {metrics['f1']:.4f}")
    print("\nConfusion matrix")
    print("----------------")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    if probabilities is not None:
        print("\nProbability distribution")
        print("------------------------")
        print(pd.Series(probabilities).describe().to_string())

    return metrics


def print_week6_targets(metrics):
    print("\nWeek 6 improvement target")
    print("-------------------------")
    print("The Week 2-style model catches failures, but creates too many false positives.")
    print("Week 6 should focus on improving precision and reducing false positives")
    print("without destroying recall.")

    print("\nBaseline to beat")
    print("----------------")
    print(f"Baseline precision: {metrics['precision']:.4f}")
    print(f"Baseline recall:    {metrics['recall']:.4f}")
    print(f"Baseline F1:        {metrics['f1']:.4f}")
    print(f"Baseline FP:        {metrics['false_positives']}")
    print(f"Baseline FN:        {metrics['false_negatives']}")

    print("\nPractical target")
    print("----------------")
    print("A useful v2 model should aim for:")
    print("- F1 higher than the Week 2 baseline")
    print("- fewer false positives")
    print("- recall near or above 40%, if possible")
    print("- a more realistic alert volume")


def main():
    df = load_feature_dataset()
    X, y = prepare_features_and_target(df)

    numeric_features, categorical_features, boolean_features = identify_feature_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = build_baseline_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        boolean_features=boolean_features,
    )

    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    print_week6_targets(metrics)


if __name__ == "__main__":
    main()
