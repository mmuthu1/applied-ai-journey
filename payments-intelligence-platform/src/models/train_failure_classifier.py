from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
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


COLUMNS_TO_EXCLUDE = [
    "payment_id",
    "payment_date",
    TARGET_COLUMN,
]


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
    print(f"Path: {path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Failure rate: {df[TARGET_COLUMN].mean():.2%}")

    return df


def prepare_features_and_target(df):
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    print("\nFeature setup")
    print("-------------")
    print(f"Number of features before encoding: {len(feature_columns)}")
    print(f"Numeric features: {len(NUMERIC_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Boolean features: {len(BOOLEAN_FEATURES)}")
    print(f"Target column: {TARGET_COLUMN}")

    return X, y


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTrain/test split")
    print("----------------")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Training failure rate: {y_train.mean():.2%}")
    print(f"Test failure rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test


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


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nModel evaluation")
    print("----------------")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 score:  {f1_score(y_test, y_pred):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\nBusiness interpretation:")
    print(f"True negatives: {tn} payments correctly predicted as not failed")
    print(f"False positives: {fp} payments flagged but did not fail")
    print(f"False negatives: {fn} failed payments missed by the model")
    print(f"True positives: {tp} failed payments correctly identified")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

def print_no_skill_baseline(y_test):
    majority_class = y_test.value_counts().idxmax()
    baseline_predictions = [majority_class] * len(y_test)

    print("\nNo-skill baseline")
    print("-----------------")
    print(f"Majority class: {majority_class}")
    print(f"Baseline accuracy: {accuracy_score(y_test, baseline_predictions):.4f}")
    print(f"Baseline recall: {recall_score(y_test, baseline_predictions):.4f}")
    print(f"Baseline F1: {f1_score(y_test, baseline_predictions):.4f}")


def train_failure_classifier():
    df = load_dataset(DATA_PATH)

    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    model = build_model_pipeline()

    print("\nTraining Logistic Regression baseline model...")
    model.fit(X_train, y_train)

    print_no_skill_baseline(y_test)
    evaluate_model(model, X_test, y_test)
    print("\nModel summary")
    print("-------------")
    print("Model type: Logistic Regression")
    print("Preprocessing: StandardScaler for numeric features, OneHotEncoder for categorical features")
    print("Class imbalance handling: class_weight='balanced'")
    print("Evaluation focus: recall and F1 score for failed-payment class")


def main():
    train_failure_classifier()


if __name__ == "__main__":
    main()
