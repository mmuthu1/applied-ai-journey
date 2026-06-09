from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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


def load_dataset(path):
    df = pd.read_csv(path)

    print("Loaded dataset")
    print("--------------")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Failure rate: {df[TARGET_COLUMN].mean():.2%}")

    return df


def prepare_features_and_target(df):
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    return X, y


def split_dataset(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


def build_preprocessor(scale_numeric=True):
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
        ]
    )


def build_logistic_regression_model(class_weight=None):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_model(class_weight="balanced"):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=10,
                    class_weight=class_weight,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_predictions(model_name, y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    missed_failure_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    operational_cost = (fp * 1) + (fn * 10)

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "false_positive_rate": false_positive_rate,
        "missed_failure_rate": missed_failure_rate,
        "operational_cost": operational_cost,
    }

    return metrics


def print_model_result(metrics):
    print("\n" + "=" * 80)
    print(metrics["model_name"])
    print("=" * 80)

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 score:  {metrics['f1']:.4f}")

    print("\nConfusion matrix values:")
    print(f"True negatives:  {metrics['true_negatives']}")
    print(f"False positives: {metrics['false_positives']}")
    print(f"False negatives: {metrics['false_negatives']}")
    print(f"True positives:  {metrics['true_positives']}")

    print("\nBusiness interpretation:")
    print(
        f"- Correctly ignored {metrics['true_negatives']} normal payments"
    )
    print(
        f"- Flagged {metrics['false_positives']} normal payments unnecessarily"
    )
    print(
        f"- Missed {metrics['false_negatives']} failed payments"
    )
    print(
        f"- Correctly identified {metrics['true_positives']} failed payments"
    )

    print(f"False positive rate: {metrics['false_positive_rate']:.4f}")

    print(f"Missed failure rate: {metrics['missed_failure_rate']:.4f}")

    print(f"Operational cost score: {metrics['operational_cost']}")


def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_predictions(model_name, y_test, y_pred)

    print_model_result(metrics)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics


def evaluate_no_skill_baseline(y_test):
    majority_class = y_test.value_counts().idxmax()
    y_pred = [majority_class] * len(y_test)

    metrics = evaluate_predictions("No-skill baseline", y_test, y_pred)

    print_model_result(metrics)

    return metrics


def print_comparison_table(results):
    comparison_df = pd.DataFrame(results)

    display_columns = [
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "false_negatives",
        "true_positives",
        "false_positive_rate",
        "missed_failure_rate",
        "operational_cost",
    ]

    comparison_df = comparison_df[display_columns]

    print("\n" + "=" * 80)
    print("Model Comparison Summary")
    print("=" * 80)
    print(comparison_df.sort_values("f1", ascending=False).to_string(index=False))


def print_recommendation(results):
    comparison_df = pd.DataFrame(results)

    best_f1_model = comparison_df.sort_values("f1", ascending=False).iloc[0]
    best_recall_model = comparison_df.sort_values("recall", ascending=False).iloc[0]

    print("\n" + "=" * 80)
    print("Operational Recommendation")
    print("=" * 80)

    print(
        f"Best F1 model: {best_f1_model['model_name']} "
        f"(F1={best_f1_model['f1']:.4f})"
    )

    print(
        f"Best recall model: {best_recall_model['model_name']} "
        f"(Recall={best_recall_model['recall']:.4f})"
    )

    print("\nInterpretation:")
    print(
        "For payment failure triage, recall matters because missed failures "
        "can create operational risk. However, too many false positives create "
        "manual review burden. The best model should balance recall with a "
        "manageable number of false positives."
    )


def main():
    df = load_dataset(DATA_PATH)

    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("\nTrain/test split")
    print("----------------")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Training failure rate: {y_train.mean():.2%}")
    print(f"Test failure rate: {y_test.mean():.2%}")

    results = []

    results.append(evaluate_no_skill_baseline(y_test))

    models = [
        (
            "Logistic Regression",
            build_logistic_regression_model(class_weight=None),
        ),
        (
            "Logistic Regression Balanced",
            build_logistic_regression_model(class_weight="balanced"),
        ),
        (
            "Random Forest Balanced",
            build_random_forest_model(class_weight="balanced"),
        ),
    ]

    for model_name, model in models:
        metrics = evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        results.append(metrics)

    print_comparison_table(results)
    print_recommendation(results)


if __name__ == "__main__":
    main()
