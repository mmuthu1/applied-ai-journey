import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


MODEL_METADATA_PATH = Path("data/monitoring/model_metadata.json")
PREDICTION_LOG_PATH = Path("data/monitoring/prediction_logs.csv")
DRIFT_SUMMARY_PATH = Path("data/monitoring/data_drift_summary.csv")

OUTPUT_PATH = Path("reports/model_monitoring_report.md")


def load_model_metadata():
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata file not found: {MODEL_METADATA_PATH}. "
            "Run python -m src.monitoring.model_metadata first."
        )

    with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_prediction_logs():
    if not PREDICTION_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Prediction log file not found: {PREDICTION_LOG_PATH}. "
            "Run python -m src.monitoring.prediction_logger first."
        )

    return pd.read_csv(PREDICTION_LOG_PATH)


def load_drift_summary():
    if not DRIFT_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Drift summary file not found: {DRIFT_SUMMARY_PATH}. "
            "Run python -m src.monitoring.data_drift_check first."
        )

    return pd.read_csv(DRIFT_SUMMARY_PATH)


def build_model_registry_section(metadata):
    rows = []

    for model in metadata["models"]:
        rows.append(
            {
                "Model": model["model_name"],
                "Version": model["model_version"],
                "Type": model["model_type"],
                "Artifact": model["artifact_path"],
                "Serving Status": model["serving_status"],
                "API Endpoint": model["api_endpoint"],
            }
        )

    model_df = pd.DataFrame(rows)

    return model_df.to_markdown(index=False)


def build_prediction_log_section(logs_df):
    if logs_df.empty:
        return (
            "No prediction events have been logged yet.\n"
        )

    events_by_model = (
        logs_df["model_name"]
        .value_counts()
        .reset_index()
    )
    events_by_model.columns = ["Model", "Prediction Events"]

    events_by_source = (
        logs_df["source"]
        .value_counts()
        .reset_index()
    )
    events_by_source.columns = ["Source", "Prediction Events"]

    latest_events = logs_df.tail(10).copy()

    return f"""
### Prediction Events by Model

{events_by_model.to_markdown(index=False)}

### Prediction Events by Source

{events_by_source.to_markdown(index=False)}

### Latest Prediction Events

{latest_events.to_markdown(index=False)}
"""


def build_drift_section(drift_df):
    drift_counts = (
        drift_df["drift_level"]
        .value_counts()
        .reset_index()
    )
    drift_counts.columns = ["Drift Level", "Feature Count"]

    high_drift_features = drift_df[
        drift_df["drift_level"] == "HIGH"
    ].copy()

    if high_drift_features.empty:
        high_drift_text = "No high-drift features detected."
    else:
        high_drift_text = high_drift_features.to_markdown(index=False)

    return f"""
### Drift Level Summary

{drift_counts.to_markdown(index=False)}

### Full Drift Summary

{drift_df.to_markdown(index=False)}

### High Drift Features

{high_drift_text}
"""


def build_governance_section():
    return """
## Governance and Monitoring Notes

This monitoring layer adds basic governance visibility to the Payments Intelligence Platform.

Current monitoring capabilities include:

- model metadata registry
- prediction event logging
- data drift summary
- model artifact tracking
- model version tracking
- known limitation documentation

These capabilities help answer operational questions such as:

- Which model version produced a prediction?
- What model artifact was used?
- What action was recommended?
- Are recent payment inputs drifting from reference data?
- Which models are available through the API?
- Which models are saved but not yet connected to the API?

## Current Limitations

This monitoring implementation is still experimental and portfolio-focused.

Current limitations:

1. Prediction logs are sample/script-generated, not yet connected to all API calls.
2. Drift checks use a simulated new scoring dataset.
3. No automated alerting exists yet.
4. No dashboard exists yet.
5. No user-level audit trail exists yet.
6. No production data is used.
7. No authentication or authorization exists.
8. No model approval workflow exists.
9. No model rollback workflow exists.
10. No real-time monitoring service exists.

## Production Readiness Recommendations

To move closer to production readiness, future work should include:

1. Connect prediction logging to API endpoints.
2. Store prediction logs in a database.
3. Schedule drift checks.
4. Add automated monitoring alerts.
5. Track model versions and deployment approvals.
6. Add model owner and reviewer metadata.
7. Add human-review outcome capture.
8. Add authentication and authorization.
9. Add a monitoring dashboard.
10. Add deployment and rollback procedures.
"""


def build_report(metadata, logs_df, drift_df):
    generated_at = datetime.now(timezone.utc).isoformat()

    model_registry_table = build_model_registry_section(metadata)
    prediction_log_section = build_prediction_log_section(logs_df)
    drift_section = build_drift_section(drift_df)
    governance_section = build_governance_section()

    report = f"""# Model Monitoring Report

Generated at:

```text
{generated_at}

Executive Summary

This report summarizes the current monitoring and governance foundation for the Payments Intelligence Platform.

The platform currently includes:

a model metadata registry
prediction event logging
data drift checks
model artifact tracking
model version tracking
known limitation documentation

This is an experimental monitoring layer for portfolio demonstration. It is not a production monitoring system yet.

Model Registry Summary

{model_registry_table}

Prediction Log Summary

Total prediction events logged: 
{len(logs_df)} 
{prediction_log_section}

Data Drift Summary

The drift check compares a reference dataset against a simulated new scoring dataset.

{drift_section}

{governance_section}

"""
    return report

def save_report(report):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        file.write(report)

    print("Saved model monitoring report")
    print("-----------------------------")
    print(f"Path: {OUTPUT_PATH}")

def main():
    metadata = load_model_metadata()
    logs_df = load_prediction_logs()
    drift_df = load_drift_summary()

    report = build_report(
        metadata=metadata,
        logs_df=logs_df,
        drift_df=drift_df,
    )

    save_report(report)

    print("\nMonitoring report summary")
    print("-------------------------")
    print(f"Models tracked: {len(metadata['models'])}")
    print(f"Prediction events logged: {len(logs_df)}")
    print(f"Drift features checked: {len(drift_df)}")
    print(
        "High drift features: "
        f"{len(drift_df[drift_df['drift_level'] == 'HIGH'])}"
    )

if __name__ == "__main__":
    main()
