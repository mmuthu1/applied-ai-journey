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
    "alert_rate": 0.385,
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


def build_model(numeric_features, categorical_features, boolean_features):
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


def evaluate_threshold(y_test, probabilities, threshold):
    predictions = probabilities >= threshold

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    metrics = {
        "threshold": threshold,
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

    metrics["operational_cost_fn10_fp1"] = (
        metrics["false_negatives"] * 10
        + metrics["false_positives"] * 1
    )

    metrics["beats_week2_f1"] = metrics["f1"] > WEEK2_BASELINE["f1"]
    metrics["reduces_week2_fp"] = (
        metrics["false_positives"] < WEEK2_BASELINE["false_positives"]
    )
    metrics["meets_40pct_recall"] = metrics["recall"] >= 0.40
    metrics["alert_rate_under_30pct"] = metrics["alert_rate"] <= 0.30

    return metrics


def tune_thresholds(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]

    results = [
        evaluate_threshold(
            y_test=y_test,
            probabilities=probabilities,
            threshold=threshold,
        )
        for threshold in thresholds
    ]

    return pd.DataFrame(results)


def print_threshold_results(results_df):
    display_columns = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "false_negatives",
        "true_positives",
        "alert_rate",
        "operational_cost_fn10_fp1",
        "beats_week2_f1",
        "reduces_week2_fp",
        "meets_40pct_recall",
        "alert_rate_under_30pct",
    ]

    display_df = results_df[display_columns].copy()

    numeric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "alert_rate",
    ]

    for column in numeric_columns:
        display_df[column] = display_df[column].round(4)

    print("\nV2 threshold tuning results")
    print("---------------------------")
    print(display_df.to_string(index=False))


def recommend_threshold(results_df):
    practical_candidates = results_df[
        (results_df["meets_40pct_recall"])
        & (results_df["alert_rate_under_30pct"])
    ].copy()

    print("\nPractical threshold candidates")
    print("------------------------------")

    if practical_candidates.empty:
        print(
            "No threshold met both recall >= 40% and alert rate <= 30%."
        )
        print(
            "Selecting best available threshold by F1 while considering alert volume."
        )

        best = results_df.sort_values(
            by=["f1", "precision"],
            ascending=False,
        ).iloc[0]
    else:
        print(practical_candidates.to_string(index=False))

        best = practical_candidates.sort_values(
            by=["f1", "precision"],
            ascending=False,
        ).iloc[0]

    print("\nRecommended v2 operating threshold")
    print("----------------------------------")
    print(f"Threshold: {best['threshold']:.2f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print(f"F1: {best['f1']:.4f}")
    print(f"False positives: {int(best['false_positives'])}")
    print(f"False negatives: {int(best['false_negatives'])}")
    print(f"True positives: {int(best['true_positives'])}")
    print(f"Alert rate: {best['alert_rate']:.2%}")
    print(f"Operational cost, FN=10 and FP=1: {int(best['operational_cost_fn10_fp1'])}")

    return best


def compare_recommended_threshold_to_week2(best):
    print("\nComparison vs Week 2 baseline")
    print("-----------------------------")
    print(f"Precision change: {best['precision'] - WEEK2_BASELINE['precision']:.4f}")
    print(f"Recall change:    {best['recall'] - WEEK2_BASELINE['recall']:.4f}")
    print(f"F1 change:        {best['f1'] - WEEK2_BASELINE['f1']:.4f}")
    print(
        "False positive change: "
        f"{int(best['false_positives'] - WEEK2_BASELINE['false_positives'])}"
    )
    print(
        "False negative change: "
        f"{int(best['false_negatives'] - WEEK2_BASELINE['false_negatives'])}"
    )
    print(f"Alert rate change: {best['alert_rate'] - WEEK2_BASELINE['alert_rate']:.4f}")


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

    model = build_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        boolean_features=boolean_features,
    )

    print("\nTraining Logistic Regression Balanced v2")
    print("----------------------------------------")
    model.fit(X_train, y_train)

    results_df = tune_thresholds(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    print_threshold_results(results_df)

    best = recommend_threshold(results_df)
    compare_recommended_threshold_to_week2(best)

    print("\nModeling note")
    print("-------------")
    print(
        "Threshold selection depends on business cost tradeoffs. "
        "If missing a failed payment is much more expensive than reviewing "
        "a normal payment, a lower threshold may be acceptable."
    )


if __name__ == "__main__":
    main()
