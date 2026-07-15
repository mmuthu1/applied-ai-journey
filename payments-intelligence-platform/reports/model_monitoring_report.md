# Model Monitoring Report

Generated at:

```text
2026-07-15T20:23:31.620434+00:00

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

| Model                         | Version   | Type              | Artifact                                 | Serving Status           | API Endpoint             |
|:------------------------------|:----------|:------------------|:-----------------------------------------|:-------------------------|:-------------------------|
| payment_failure_classifier_v1 | v1        | classification    | models/payment_failure_classifier.pkl    | available_via_api        | /predict/payment-failure |
| payment_failure_classifier_v2 | v2        | classification    | models/payment_failure_classifier_v2.pkl | saved_artifact_available | not_yet_connected        |
| cash_forecast_model           | v1        | regression        | models/cash_forecast_model.pkl           | available_via_api        | /predict/cash-forecast   |
| payment_anomaly_detector      | v1        | anomaly_detection | models/payment_anomaly_detector.pkl      | available_via_api        | /predict/payment-anomaly |

Prediction Log Summary

Total prediction events logged: 
7 

### Prediction Events by Model

| Model                         |   Prediction Events |
|:------------------------------|--------------------:|
| payment_failure_classifier_v2 |                   5 |
| cash_forecast_model           |                   1 |
| payment_anomaly_detector      |                   1 |

### Prediction Events by Source

| Source                    |   Prediction Events |
|:--------------------------|--------------------:|
| sample_monitoring_script  |                   4 |
| predict_failure_v2_script |                   3 |

### Latest Prediction Events

| logged_at                        | model_name                    | model_version   | model_type        | record_id   |   prediction_value |   prediction_probability | prediction_band   | recommended_action                                     | source                    |
|:---------------------------------|:------------------------------|:----------------|:------------------|:------------|-------------------:|-------------------------:|:------------------|:-------------------------------------------------------|:--------------------------|
| 2026-07-15T16:32:35.047626+00:00 | payment_failure_classifier_v2 | v2              | classification    | NEW-V2-1001 |          0         |                   0.1537 | LOW               | Allow normal processing                                | sample_monitoring_script  |
| 2026-07-15T16:32:35.048101+00:00 | payment_failure_classifier_v2 | v2              | classification    | NEW-V2-1002 |          1         |                   0.7424 | HIGH              | Review before release                                  | sample_monitoring_script  |
| 2026-07-15T16:32:35.048105+00:00 | cash_forecast_model           | v1              | regression        | 2024-06-28  |          3.111e+06 |                 nan      | MEDIUM            | Monitor expected payment activity and exception volume | sample_monitoring_script  |
| 2026-07-15T16:32:35.048108+00:00 | payment_anomaly_detector      | v1              | anomaly_detection | NEW-A1002   |         -0.1636    |                 nan      | HIGH              | High-priority anomaly review                           | sample_monitoring_script  |
| 2026-07-15T20:16:52.596981+00:00 | payment_failure_classifier_v2 | v2              | classification    | NEW-V2-1001 |          0         |                   0.1537 | LOW               | Allow normal processing                                | predict_failure_v2_script |
| 2026-07-15T20:16:52.602164+00:00 | payment_failure_classifier_v2 | v2              | classification    | NEW-V2-1002 |          1         |                   0.7424 | HIGH              | Review before release                                  | predict_failure_v2_script |
| 2026-07-15T20:16:52.604519+00:00 | payment_failure_classifier_v2 | v2              | classification    | NEW-V2-1003 |          0         |                   0.4477 | MEDIUM            | Monitor or queue for secondary review                  | predict_failure_v2_script |


Data Drift Summary

The drift check compares a reference dataset against a simulated new scoring dataset.


### Drift Level Summary

| Drift Level   |   Feature Count |
|:--------------|----------------:|
| HIGH          |               6 |
| LOW           |               2 |

### Full Drift Summary

| feature                  | feature_type   | reference_value   | new_value   | comparison_metric   |   change_pct | drift_level   |   reference_median |   new_median |
|:-------------------------|:---------------|:------------------|:------------|:--------------------|-------------:|:--------------|-------------------:|-------------:|
| amount                   | numeric        | 125237.8028       | 167306.0454 | mean                |        33.59 | HIGH          |          60695.5   |     89429.5  |
| counterparty_risk_score  | numeric        | 0.2879            | 0.3689      | mean                |        28.13 | HIGH          |              0.266 |         0.34 |
| historical_failure_count | numeric        | 1.0012            | 0.99        | mean                |        -1.12 | LOW           |              1     |         1    |
| risk_adjusted_amount     | numeric        | 36310.2492        | 61822.9597  | mean                |        70.26 | HIGH          |          14411.9   |     29937.2  |
| payment_type             | categorical    | ACH               | SWIFT       | top_category_share  |         4.06 | HIGH          |            nan     |       nan    |
| channel                  | categorical    | BATCH             | FILE        | top_category_share  |         4.22 | HIGH          |            nan     |       nan    |
| country                  | categorical    | JP                | IN          | top_category_share  |         0.68 | HIGH          |            nan     |       nan    |
| currency                 | categorical    | CAD               | CAD         | top_category_share  |         0.92 | LOW           |            nan     |       nan    |

### High Drift Features

| feature                 | feature_type   | reference_value   | new_value   | comparison_metric   |   change_pct | drift_level   |   reference_median |   new_median |
|:------------------------|:---------------|:------------------|:------------|:--------------------|-------------:|:--------------|-------------------:|-------------:|
| amount                  | numeric        | 125237.8028       | 167306.0454 | mean                |        33.59 | HIGH          |          60695.5   |     89429.5  |
| counterparty_risk_score | numeric        | 0.2879            | 0.3689      | mean                |        28.13 | HIGH          |              0.266 |         0.34 |
| risk_adjusted_amount    | numeric        | 36310.2492        | 61822.9597  | mean                |        70.26 | HIGH          |          14411.9   |     29937.2  |
| payment_type            | categorical    | ACH               | SWIFT       | top_category_share  |         4.06 | HIGH          |            nan     |       nan    |
| channel                 | categorical    | BATCH             | FILE        | top_category_share  |         4.22 | HIGH          |            nan     |       nan    |
| country                 | categorical    | JP                | IN          | top_category_share  |         0.68 | HIGH          |            nan     |       nan    |



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


