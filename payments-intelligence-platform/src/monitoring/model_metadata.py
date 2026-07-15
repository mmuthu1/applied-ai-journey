import json
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_PATH = Path("data/monitoring/model_metadata.json")


def build_model_metadata():
    generated_at = datetime.now(timezone.utc).isoformat()

    metadata = {
        "generated_at": generated_at,
        "project": "Payments Intelligence Platform",
        "environment": "local_development",
        "models": [
            {
                "model_name": "payment_failure_classifier_v1",
                "model_version": "v1",
                "model_type": "classification",
                "artifact_path": "models/payment_failure_classifier.pkl",
                "training_dataset": "data/processed/payments_features.csv",
                "target_column": "is_failed",
                "serving_status": "available_via_api",
                "api_endpoint": "/predict/payment-failure",
                "recommended_threshold": 0.50,
                "primary_metrics": {
                    "precision": 0.0779,
                    "recall": 0.5357,
                    "f1": 0.1361,
                    "false_positives": 355,
                    "false_negatives": 26,
                },
                "known_limitations": [
                    "Trained on synthetic data",
                    "High false-positive rate",
                    "Not production-ready",
                    "No prediction logging in v1 API flow yet",
                ],
            },
            {
                "model_name": "payment_failure_classifier_v2",
                "model_version": "v2",
                "model_type": "classification",
                "artifact_path": "models/payment_failure_classifier_v2.pkl",
                "training_dataset": "data/processed/payments_failure_features_v2.csv",
                "target_column": "is_failed",
                "serving_status": "saved_artifact_available",
                "api_endpoint": "not_yet_connected",
                "recommended_threshold": 0.55,
                "primary_metrics": {
                    "precision": 0.1049,
                    "recall": 0.5000,
                    "f1": 0.1734,
                    "false_positives": 239,
                    "false_negatives": 28,
                    "alert_rate": 0.267,
                },
                "known_limitations": [
                    "Trained on synthetic data",
                    "Historical failure-rate features are dataset-level approximations",
                    "Production version needs time-aware historical features",
                    "Not yet connected to FastAPI endpoint",
                    "No prediction logging in v2 inference flow yet",
                ],
            },
            {
                "model_name": "cash_forecast_model",
                "model_version": "v1",
                "model_type": "regression",
                "artifact_path": "models/cash_forecast_model.pkl",
                "training_dataset": "data/processed/cash_forecast_daily.csv",
                "target_column": "next_day_total_amount",
                "serving_status": "available_via_api",
                "api_endpoint": "/predict/cash-forecast",
                "recommended_threshold": None,
                "primary_metrics": {
                    "mae": 894179.20,
                    "rmse": 1163787.60,
                    "mape": 0.3000,
                    "improvement_vs_baseline_mae_pct": 0.0322,
                },
                "known_limitations": [
                    "Trained on synthetic data",
                    "Only 173 daily records",
                    "No prediction intervals",
                    "Not production-ready",
                ],
            },
            {
                "model_name": "payment_anomaly_detector",
                "model_version": "v1",
                "model_type": "anomaly_detection",
                "artifact_path": "models/payment_anomaly_detector.pkl",
                "training_dataset": "data/processed/payment_anomaly_features.csv",
                "target_column": None,
                "serving_status": "available_via_api",
                "api_endpoint": "/predict/payment-anomaly",
                "recommended_threshold": None,
                "primary_metrics": {
                    "contamination": 0.04,
                    "combined_anomaly_rate": 0.0484,
                    "rule_and_model_count": 141,
                    "model_only_count": 59,
                    "rule_only_count": 42,
                },
                "known_limitations": [
                    "Trained on synthetic data",
                    "Unsupervised anomaly labels are not confirmed fraud or failure labels",
                    "Requires human review workflow",
                    "Not production-ready",
                ],
            },
        ],
    }

    return metadata


def save_model_metadata(metadata):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Saved model metadata registry")
    print("-----------------------------")
    print(f"Path: {OUTPUT_PATH}")
    print(f"Model count: {len(metadata['models'])}")
    print(f"Generated at: {metadata['generated_at']}")


def print_model_summary(metadata):
    print("\nModel registry summary")
    print("----------------------")

    for model in metadata["models"]:
        print(
            f"{model['model_name']} | "
            f"version={model['model_version']} | "
            f"type={model['model_type']} | "
            f"status={model['serving_status']}"
        )


def main():
    metadata = build_model_metadata()
    save_model_metadata(metadata)
    print_model_summary(metadata)


if __name__ == "__main__":
    main()
