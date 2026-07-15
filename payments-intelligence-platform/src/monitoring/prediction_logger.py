from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


LOG_PATH = Path("data/monitoring/prediction_logs.csv")


LOG_COLUMNS = [
    "logged_at",
    "model_name",
    "model_version",
    "model_type",
    "record_id",
    "prediction_value",
    "prediction_probability",
    "prediction_band",
    "recommended_action",
    "source",
]


def build_prediction_log_event(
    model_name,
    model_version,
    model_type,
    record_id,
    prediction_value,
    prediction_probability=None,
    prediction_band=None,
    recommended_action=None,
    source="manual_script",
):
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "model_type": model_type,
        "record_id": record_id,
        "prediction_value": prediction_value,
        "prediction_probability": prediction_probability,
        "prediction_band": prediction_band,
        "recommended_action": recommended_action,
        "source": source,
    }


def ensure_log_file_exists():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not LOG_PATH.exists():
        empty_log_df = pd.DataFrame(columns=LOG_COLUMNS)
        empty_log_df.to_csv(LOG_PATH, index=False)


def append_prediction_log(event):
    ensure_log_file_exists()

    missing_columns = [
        column for column in LOG_COLUMNS if column not in event
    ]

    if missing_columns:
        raise ValueError(f"Missing log event columns: {missing_columns}")

    event_df = pd.DataFrame([event], columns=LOG_COLUMNS)
    event_df.to_csv(LOG_PATH, mode="a", header=False, index=False)


def append_prediction_logs(events):
    ensure_log_file_exists()

    for event in events:
        missing_columns = [
            column for column in LOG_COLUMNS if column not in event
        ]

        if missing_columns:
            raise ValueError(f"Missing log event columns: {missing_columns}")

    events_df = pd.DataFrame(events, columns=LOG_COLUMNS)
    events_df.to_csv(LOG_PATH, mode="a", header=False, index=False)


def load_prediction_logs():
    ensure_log_file_exists()
    return pd.read_csv(LOG_PATH)


def summarize_prediction_logs(logs_df):
    print("Prediction log summary")
    print("----------------------")
    print(f"Log path: {LOG_PATH}")
    print(f"Total prediction events: {len(logs_df)}")

    if logs_df.empty:
        print("No prediction events logged yet.")
        return

    print("\nEvents by model")
    print("---------------")
    print(logs_df["model_name"].value_counts().to_string())

    print("\nEvents by source")
    print("----------------")
    print(logs_df["source"].value_counts().to_string())

    print("\nLatest events")
    print("-------------")
    print(logs_df.tail(10).to_string(index=False))


def create_sample_log_events():
    return [
        build_prediction_log_event(
            model_name="payment_failure_classifier_v2",
            model_version="v2",
            model_type="classification",
            record_id="NEW-V2-1001",
            prediction_value=0,
            prediction_probability=0.1537,
            prediction_band="LOW",
            recommended_action="Allow normal processing",
            source="sample_monitoring_script",
        ),
        build_prediction_log_event(
            model_name="payment_failure_classifier_v2",
            model_version="v2",
            model_type="classification",
            record_id="NEW-V2-1002",
            prediction_value=1,
            prediction_probability=0.7424,
            prediction_band="HIGH",
            recommended_action="Review before release",
            source="sample_monitoring_script",
        ),
        build_prediction_log_event(
            model_name="cash_forecast_model",
            model_version="v1",
            model_type="regression",
            record_id="2024-06-28",
            prediction_value=3110997.77,
            prediction_probability=None,
            prediction_band="MEDIUM",
            recommended_action="Monitor expected payment activity and exception volume",
            source="sample_monitoring_script",
        ),
        build_prediction_log_event(
            model_name="payment_anomaly_detector",
            model_version="v1",
            model_type="anomaly_detection",
            record_id="NEW-A1002",
            prediction_value=-0.1636,
            prediction_probability=None,
            prediction_band="HIGH",
            recommended_action="High-priority anomaly review",
            source="sample_monitoring_script",
        ),
    ]


def main():
    print("Creating sample prediction log events...")
    events = create_sample_log_events()

    append_prediction_logs(events)

    logs_df = load_prediction_logs()
    summarize_prediction_logs(logs_df)


if __name__ == "__main__":
    main()
