# Model Monitoring Review Runbook

## Purpose

This runbook explains how to review model monitoring outputs for the Payments Intelligence Platform.

The monitoring layer tracks model metadata, prediction logs, data drift, and production-readiness gaps.

## Monitoring Components

The current monitoring layer includes:

- model metadata registry
- prediction audit log
- data drift summary
- generated monitoring report

## Model Metadata Registry

Generated file:

```text
data/monitoring/model_metadata.json
```

The registry tracks:

    model name
    model version
    model type
    artifact path
    training dataset
    target column
    serving status
    API endpoint
    recommended threshold
    primary metrics
    known limitations

Operations and governance teams can use this to understand which models exist, which versions are available, and which artifacts are being used.

## Prediction Audit Log

Generated file:

```text
data/monitoring/prediction_logs.csv
```

The prediction log tracks:

    timestamp
    model name
    model version
    model type
    record ID
    prediction value
    prediction probability or score
    prediction band
    recommended action
    source

This supports traceability and basic auditability.

## Data Drift Summary

Generated file:

```text
data/monitoring/data_drift_summary.csv
```

The drift summary compares reference data with simulated new scoring data.

Features checked include:

    amount
    counterparty risk score
    historical failure count
    risk-adjusted amount
    payment type
    channel
    country
    currency

## Drift Levels

### LOW Drift

Meaning:

```text
No major change detected
```

Recommended action:

```text
Continue normal monitoring
```

### MEDIUM Drift

Meaning:

```text
Moderate change detected
```

Recommended action:

```text
Review feature distribution and monitor closely
```

### HIGH Drift

Meaning:

```text
Significant change detected
```

Recommended action:

```text
Investigate data shift before relying on model outputs
```

Operations guidance for HIGH drift:

    Identify which feature drifted.
    Check whether the change is expected.
    Compare recent payment activity against historical patterns.
    Review whether business volume, client mix, channel mix, or country mix changed.
    Consider whether model performance may be affected.
    Escalate to model owner if drift persists.

## Monitoring Report

Generated file:

```text
reports/model_monitoring_report.md
```

The report summarizes:

    model registry status
    prediction log activity
    drift results
    governance notes
    limitations
    production-readiness recommendations

## Current Limitations

The monitoring layer is not production-ready.

Current limitations:

1. Prediction logs are file-based.
2. Drift checks use simulated new scoring data.
3. No automated alerting exists.
4. No monitoring dashboard exists.
5. No scheduled monitoring jobs exist.
6. No model approval workflow exists.
7. No rollback workflow exists.
8. No production data is used.

## Production Recommendations
Before production use, the following improvements are required:

1. Store logs in a database.
2. Schedule drift checks.
3. Add automated alerts.
4. Add monitoring dashboard.
5. Add model owner and reviewer metadata.
6. Add model approval workflow.
7. Add rollback process.
8. Capture human review outcomes.
9. Compare predictions against actual outcomes.