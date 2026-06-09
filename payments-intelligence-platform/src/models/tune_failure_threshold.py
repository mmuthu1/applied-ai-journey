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


THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
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


def build_balanced_logistic_regression():
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


def calculate_metrics_for_threshold(y_test, y_prob, threshold):
    y_pred = y_prob >= threshold
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    missed_failure_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    operational_cost_10x = (fp * 1) + (fn * 10)
    operational_cost_25x = (fp * 1) + (fn * 25)
    operational_cost_50x = (fp * 1) + (fn * 50)

    alert_volume = fp + tp
    alert_rate = (fp + tp) / len(y_test)


    return {
        "threshold": threshold,
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
        "operational_cost_10x": operational_cost_10x,
        "operational_cost_25x": operational_cost_25x,
        "operational_cost_50x": operational_cost_50x,
        "alert_volume": alert_volume,
        "alert_rate": alert_rate,
    }


def evaluate_thresholds(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]

    results = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics_for_threshold(
            y_test=y_test,
            y_prob=y_prob,
            threshold=threshold,
        )

        results.append(metrics)

    return pd.DataFrame(results)


def print_threshold_comparison(results_df):
    display_columns = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "false_negatives",
        "true_positives",
        "false_positive_rate",
        "missed_failure_rate",
        "operational_cost_10x",
        "operational_cost_25x",
        "operational_cost_50x",
        "alert_volume",
        "alert_rate",
    ]

    print("\nThreshold Comparison")
    print("--------------------")
    print(results_df[display_columns].to_string(index=False))


def print_recommendations(results_df):
    best_f1 = results_df.sort_values("f1", ascending=False).iloc[0]
    best_cost_10x = results_df.sort_values("operational_cost_10x").iloc[0]
    best_cost_25x = results_df.sort_values("operational_cost_25x").iloc[0]
    best_cost_50x = results_df.sort_values("operational_cost_50x").iloc[0]

    print("\nThreshold Recommendations")
    print("-------------------------")
    print(
        f"Best F1 threshold: {best_f1['threshold']:.2f} "
        f"(F1={best_f1['f1']:.4f}, Recall={best_f1['recall']:.4f}, "
        f"Precision={best_f1['precision']:.4f})"
    )

    print(
        f"Best cost threshold, FN cost 10x: {best_cost_10x['threshold']:.2f} "
        f"(Cost={best_cost_10x['operational_cost_10x']:.0f})"
    )

    print(
        f"Best cost threshold, FN cost 25x: {best_cost_25x['threshold']:.2f} "
        f"(Cost={best_cost_25x['operational_cost_25x']:.0f})"
    )

    print(
        f"Best cost threshold, FN cost 50x: {best_cost_50x['threshold']:.2f} "
        f"(Cost={best_cost_50x['operational_cost_50x']:.0f})"
    )

    print("\nBusiness interpretation:")
    print(
        "Lower thresholds catch more failed payments but create more operational alerts. "
        "Higher thresholds reduce alert volume but miss more failed payments. "
        "The right threshold depends on how expensive a missed failure is compared with "
        "a false alert."
    )

def print_practical_recommendation(results_df):
    practical_options = results_df[
        (results_df["recall"] >= 0.40)
        & (results_df["alert_rate"] <= 0.30)
    ]

    print("\nPractical Triage Recommendation")
    print("-------------------------------")

    if len(practical_options) == 0:
        print(
            "No threshold met the practical rule: recall >= 40% and alert_rate <= 30%."
        )
        print(
            "This means the current model is not yet strong enough for a practical triage queue."
        )
        return

    best_option = practical_options.sort_values("f1", ascending=False).iloc[0]

    print(
        f"Recommended threshold: {best_option['threshold']:.2f}"
    )
    print(
        f"Recall: {best_option['recall']:.4f}"
    )
    print(
        f"Precision: {best_option['precision']:.4f}"
    )
    print(
        f"Alert rate: {best_option['alert_rate']:.2%}"
    )
    print(
        f"False positives: {best_option['false_positives']:.0f}"
    )
    print(
        f"False negatives: {best_option['false_negatives']:.0f}"
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

    model = build_balanced_logistic_regression()

    print("\nTraining balanced Logistic Regression model...")
    model.fit(X_train, y_train)

    results_df = evaluate_thresholds(model, X_test, y_test)

    print_threshold_comparison(results_df)
    print_recommendations(results_df)
    print_practical_recommendation(results_df)

if __name__ == "__main__":
    main()
