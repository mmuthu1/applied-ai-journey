from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


def load_dataset():
    df = pd.read_csv(DATA_PATH)

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


def build_logistic_regression_pipeline():
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_pipeline():
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=10,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def get_feature_names_from_pipeline(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]

    numeric_names = NUMERIC_FEATURES

    categorical_transformer = preprocessor.named_transformers_["categorical"]
    categorical_names = list(
        categorical_transformer.get_feature_names_out(CATEGORICAL_FEATURES)
    )

    boolean_names = BOOLEAN_FEATURES

    return numeric_names + categorical_names + boolean_names


def analyze_logistic_regression_coefficients(model, X_train, y_train):
    print("\n" + "=" * 80)
    print("Logistic Regression Coefficients")
    print("=" * 80)

    model.fit(X_train, y_train)

    feature_names = get_feature_names_from_pipeline(model)
    coefficients = model.named_steps["model"].coef_[0]

    coefficient_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": abs(coefficients),
        }
    )

    print("\nTop features pushing toward FAILED prediction:")
    print(
        coefficient_df.sort_values("coefficient", ascending=False)
        .head(15)
        [["feature", "coefficient"]]
        .to_string(index=False)
    )

    print("\nTop features pushing toward NOT FAILED prediction:")
    print(
        coefficient_df.sort_values("coefficient", ascending=True)
        .head(15)
        [["feature", "coefficient"]]
        .to_string(index=False)
    )

    print("\nTop features by absolute coefficient size:")
    print(
        coefficient_df.sort_values("absolute_coefficient", ascending=False)
        .head(15)
        [["feature", "coefficient", "absolute_coefficient"]]
        .to_string(index=False)
    )

    return model, coefficient_df


def analyze_random_forest_importance(model, X_train, y_train):
    print("\n" + "=" * 80)
    print("Random Forest Feature Importance")
    print("=" * 80)

    model.fit(X_train, y_train)

    feature_names = get_feature_names_from_pipeline(model)
    importances = model.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    print("\nTop Random Forest feature importances:")
    print(
        importance_df.sort_values("importance", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    return model, importance_df


def analyze_top_predicted_risk_payments(model, X_test):
    print("\n" + "=" * 80)
    print("Top Predicted Failure-Risk Payments")
    print("=" * 80)

    predicted_probabilities = model.predict_proba(X_test)[:, 1]

    risk_df = X_test.copy()
    risk_df["predicted_failure_probability"] = predicted_probabilities

    top_risk_payments = risk_df.sort_values(
        "predicted_failure_probability",
        ascending=False,
    ).head(15)

    display_columns = [
        "predicted_failure_probability",
        "amount",
        "currency",
        "country",
        "payment_type",
        "channel",
        "counterparty_risk_score",
        "historical_failure_count",
        "amount_bucket",
        "risk_band",
        "risk_adjusted_amount",
    ]

    print(top_risk_payments[display_columns].to_string(index=False))


def print_model_signal_interpretation():
    print("\n" + "=" * 80)
    print("Model Signal Interpretation")
    print("=" * 80)

    print(
        "1. Logistic Regression coefficients show direction: positive values push "
        "toward FAILED, while negative values push toward NOT FAILED."
    )
    print(
        "2. Random Forest feature importance shows which features were most useful "
        "for splitting records across decision trees."
    )
    print(
        "3. If model signals align with EDA findings, confidence improves. For example, "
        "SWIFT, FILE, high-value, prior failures, and risk-adjusted amount should appear "
        "as meaningful signals."
    )
    print(
        "4. If model signals do not align with business intuition, investigate data quality, "
        "feature design, leakage, or weak signal strength."
    )


def print_improvement_recommendations():
    print("\n" + "=" * 80)
    print("Model Improvement Recommendations")
    print("=" * 80)

    recommendations = [
        "Add stronger historical counterparty-level features.",
        "Create rolling failure-rate features by payment_type, channel, country, and counterparty.",
        "Try additional models such as Gradient Boosting later.",
        "Tune class weights and thresholds together.",
        "Create a model evaluation report comparing recall, precision, F1, alert volume, and operational cost.",
        "Consider using probability bands for triage: high, medium, and low risk.",
    ]

    for item in recommendations:
        print(f"- {item}")


def main():
    df = load_dataset()

    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    logistic_model = build_logistic_regression_pipeline()
    logistic_model, coefficient_df = analyze_logistic_regression_coefficients(
        logistic_model,
        X_train,
        y_train,
    )

    random_forest_model = build_random_forest_pipeline()
    random_forest_model, importance_df = analyze_random_forest_importance(
        random_forest_model,
        X_train,
        y_train,
    )

    analyze_top_predicted_risk_payments(logistic_model, X_test)

    print_model_signal_interpretation()
    print_improvement_recommendations()


if __name__ == "__main__":
    main()
