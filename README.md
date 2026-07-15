# Applied AI Journey

This repository tracks my hands-on journey toward applied AI engineering leadership, with a focus on financial technology, payment operations, machine learning, model serving, MLOps, and future GenAI systems.

## Current Project

### Payments Intelligence Platform

Location:

```text
payments-intelligence-platform/
```

The Payments Intelligence Platform is an applied AI/ML engineering project focused on:

* payment failure prediction
* cash forecasting
* payment anomaly detection
* API-based model serving
* operations intelligence
* future GenAI/RAG support for payment operations

## Current Platform Capabilities

The project currently includes:

The project currently includes:

* synthetic payment data generation
* raw and clean data validation
* feature engineering
* payment failure prediction model
* cash forecasting model
* payment anomaly detection workflow
* saved model artifacts
* FastAPI model-serving endpoints
* automated data quality tests
* automated API endpoint tests
* payment failure classifier v2 improvement
* model metadata tracking
* prediction audit logging
* data drift monitoring
* generated model monitoring report

Current test status:

```text
22 passed
```

## Project Progress

### Week 1: Data Foundation — Completed

Week 1 completed the data foundation for the Payments Intelligence Platform:

* synthetic payment data generation
* raw data validation
* data cleaning
* feature engineering
* exploratory failure-pattern analysis
* end-to-end pipeline orchestration
* data quality tests with pytest

### Week 2: Payment Failure Classifier — Completed

Week 2 completed the first applied ML model lifecycle:

* baseline payment failure classifier
* model comparison across Logistic Regression and Random Forest
* threshold tuning for operational triage
* model signal and feature importance analysis
* model evaluation report
* saved model artifact
* inference script for scoring new payment records

The current model is an experimental baseline. It is useful for learning and portfolio demonstration, but it is not yet production-ready.

### Week 3: Cash Forecasting — Completed

Week 3 added a second applied AI use case to the Payments Intelligence Platform.

Completed work:

* daily cash forecasting dataset
* baseline forecasts using previous day, 3-day average, and 7-day average
* time-based train/test split
* Linear Regression and Random Forest regression models
* forecast evaluation using MAE, RMSE, and MAPE
* cash forecasting model evaluation report
* saved cash forecasting model artifact
* inference script for scoring new daily cash activity records

The current best experimental forecasting model is a Random Forest Regressor. It improved MAE by 3.22% compared with the 7-day moving average baseline.

The model is useful for learning and portfolio demonstration, but it is not yet production-ready.

### Week 4: Payment Anomaly Detection — Completed

Week 4 added a third applied AI use case to the Payments Intelligence Platform.

Completed work:

* rule-based payment anomaly features
* amount and risk-adjusted anomaly scoring
* Isolation Forest anomaly detection
* rule vs model anomaly comparison
* anomaly source classification:

  * `RULE_AND_MODEL`
  * `MODEL_ONLY`
  * `RULE_ONLY`
  * `NORMAL`
* review priority queues:

  * `P1_HIGH`
  * `P2_INVESTIGATE`
  * `P3_KNOWN_RISK`
  * `P4_NORMAL`
* payment anomaly detection report
* saved anomaly model artifact
* inference script for scoring new payment records

The anomaly workflow combines business-rule detection with unsupervised machine learning. The current workflow is useful for learning and portfolio demonstration, but it is not yet production-ready.

### Week 5: FastAPI Model Serving — Completed

Week 5 turned the platform from standalone ML scripts into an API-backed application.

Completed work:

* health check endpoint
* model metadata endpoint
* payment failure prediction endpoint
* cash forecast prediction endpoint
* payment anomaly detection endpoint
* automated API tests

Available API endpoints:

```text
GET  /
GET  /health
GET  /models

POST /predict/payment-failure
POST /predict/cash-forecast
POST /predict/payment-anomaly
```

Test status:

```text
22 passed
```

Key outcome:

```text
Turned standalone ML inference scripts into API-backed model-serving endpoints.
```

### Week 6: Payment Failure Classifier v2 — Completed

Week 6 improved the payment failure classifier by adding stronger failure-risk features and tuning the model threshold for a better operational tradeoff.

Completed work:

- diagnosed the Week 2 baseline model weakness
- created v2 failure-risk features
- trained v2 model candidates
- tuned the v2 operating threshold
- saved a v2 model artifact
- created a v2 inference script
- documented the v2 model evaluation

Recommended v2 setup:

```text
Model: Logistic Regression Balanced v2
Threshold: 0.55
```
Key improvement:

```text
Reduced false positives from 355 to 239 while keeping recall at 50%.
```

### Week 7: Monitoring, Governance, and MLOps Foundations — Completed

Week 7 added a basic monitoring and governance layer to make the platform more operationally realistic.

Completed work:

* model metadata registry
* model version tracking
* model artifact tracking
* prediction audit logging
* data drift monitoring check
* generated model monitoring report
* governance notes
* production-readiness recommendations

Key files added:

```text
src/monitoring/model_metadata.py
src/monitoring/prediction_logger.py
src/monitoring/data_drift_check.py
src/monitoring/generate_monitoring_report.py

data/monitoring/model_metadata.json
data/monitoring/prediction_logs.csv
data/monitoring/data_drift_summary.csv

reports/model_monitoring_report.md
```
Key outcome:

```text
Added monitoring and governance foundations for model metadata, prediction auditability, drift visibility, and production-readiness planning.
```
The monitoring layer is experimental and portfolio-focused. It is not yet a production MLOps system.

## Current Status

The Payments Intelligence Platform now includes an end-to-end applied AI workflow covering data pipelines, machine learning models, model artifacts, FastAPI serving, automated tests, model improvement, and monitoring/governance foundations.

Current status:

```text
Experimental applied AI platform for payment operations
```
Not yet:

```text
Production-ready financial technology system
```


## Next Step

The next project phase will focus on GenAI/RAG-based operations support.

Planned next work:

* payment operations runbooks
* retrieval over project documentation and runbooks
* AI assistant for payment operations questions
* explanation support for flagged payments
* anomaly review support
* cash forecast interpretation support
* guardrails and limitations documentation