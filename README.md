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

## Current Status

The Payments Intelligence Platform currently demonstrates an end-to-end applied AI workflow:

```text
data generation → validation → feature engineering → model training → evaluation → saved artifacts → inference → FastAPI serving → automated tests
```

The project is still experimental and uses synthetic payment operations data. It is designed for learning and portfolio demonstration, not production use.

## Next Step

The next project phase will focus on improving the payment failure model.

Planned Week 6 work:

* add stronger counterparty-level historical features
* add rolling failure-rate features
* improve model signal quality
* reduce false positives
* compare improved model performance against the Week 2 baseline
* update the model evaluation report
* prepare a short LinkedIn progress update
