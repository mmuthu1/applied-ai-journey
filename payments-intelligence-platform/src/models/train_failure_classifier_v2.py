from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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


DATA_PATH = Path("data/processed/payments_failure_features_v2.csv")

TARGET_COLUMN = "is_failed"

ID_COLUMNS = [
    "payment_id",
    "payment_date",
]

LEAKAGE_COLUMNS = [
    "status",
    "failure_reason",
]

WEEK2_BASELINE = {
    "precision": 0.0779,
    "recall": 0.5357,
    "f1": 0.1361,
    "false_positives": 355,
    "false_negatives": 26,
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


def build_preprocessor(numeric_features, categorical_features, boolean_features):
    return ColumnTransformer(
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


def build_models(preprocessor):
    return {
        "Logistic Regression Balanced v2": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest Balanced v2": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Gradient Boosting v2": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = None

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "alert_rate": (fp + tp) / len(y_test),
    }

    if probabilities is not None:
        metrics["avg_probability"] = probabilities.mean()
        metrics["max_probability"] = probabilities.max()
    else:
        metrics["avg_probability"] = None
        metrics["max_probability"] = None

    return metrics


def compare_to_week2_baseline(results_df):
    comparison_df = results_df.copy()

    comparison_df["f1_change_vs_week2"] = (
        comparison_df["f1"] - WEEK2_BASELINE["f1"]
    )

    comparison_df["precision_change_vs_week2"] = (
        comparison_df["precision"] - WEEK2_BASELINE["precision"]
    )

    comparison_df["recall_change_vs_week2"] = (
        comparison_df["recall"] - WEEK2_BASELINE["recall"]
    )

    comparison_df["false_positive_change_vs_week2"] = (
        comparison_df["false_positives"] - WEEK2_BASELINE["false_positives"]
    )

    comparison_df["false_negative_change_vs_week2"] = (
        comparison_df["false_negatives"] - WEEK2_BASELINE["false_negatives"]
    )

    return comparison_df


def print_results(results_df):
    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "alert_rate",
        "f1_change_vs_week2",
        "false_positive_change_vs_week2",
    ]

    display_df = results_df[display_columns].copy()

    numeric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "alert_rate",
        "f1_change_vs_week2",
    ]

    for column in numeric_columns:
        display_df[column] = display_df[column].round(4)

    print("\nV2 model comparison")
    print("-------------------")
    print(display_df.to_string(index=False))


def select_best_model(results_df):
    candidate_df = results_df.copy()

    candidate_df["meets_recall_target"] = candidate_df["recall"] >= 0.40
    candidate_df["beats_week2_f1"] = candidate_df["f1"] > WEEK2_BASELINE["f1"]
    candidate_df["reduces_false_positives"] = (
        candidate_df["false_positives"] < WEEK2_BASELINE["false_positives"]
    )

    print("\nV2 target checks")
    print("----------------")
    print(
        candidate_df[
            [
                "model",
                "meets_recall_target",
                "beats_week2_f1",
                "reduces_false_positives",
            ]
        ].to_string(index=False)
    )

    best_by_f1 = candidate_df.sort_values(
        by=["f1", "precision"],
        ascending=False,
    ).iloc[0]

    print("\nBest v2 candidate by F1")
    print("-----------------------")
    print(f"Model: {best_by_f1['model']}")
    print(f"Precision: {best_by_f1['precision']:.4f}")
    print(f"Recall: {best_by_f1['recall']:.4f}")
    print(f"F1: {best_by_f1['f1']:.4f}")
    print(f"False positives: {int(best_by_f1['false_positives'])}")
    print(f"False negatives: {int(best_by_f1['false_negatives'])}")
    print(f"Alert rate: {best_by_f1['alert_rate']:.2%}")

    return best_by_f1


def main():
    df = load_v2_dataset()
    X, y = prepare_features_and_target(df)

    numeric_features, categorical_features, boolean_features = identify_feature_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTrain/test split")
    print("----------------")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Train failure rate: {y_train.mean():.2%}")
    print(f"Test failure rate: {y_test.mean():.2%}")

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        boolean_features=boolean_features,
    )

    models = build_models(preprocessor)

    results = []

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")
        metrics = evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(metrics)

    results_df = pd.DataFrame(results)
    comparison_df = compare_to_week2_baseline(results_df)

    print_results(comparison_df)
    select_best_model(comparison_df)

    print("\nImportant modeling note")
    print("-----------------------")
    print(
        "The v2 dataset includes dataset-level failure-rate features. "
        "For this portfolio project, they are useful for experimentation. "
        "In production, they must be computed using only historical data "
        "available before prediction time."
    )


if __name__ == "__main__":
    main()
