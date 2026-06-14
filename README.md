# Applied AI Journey

This repository tracks my hands-on journey toward applied AI engineering leadership, with a focus on financial technology, payment operations, machine learning, and GenAI systems.

## Current Project

### Payments Intelligence Platform

Location:

```text
payments-intelligence-platform/
```

The Payments Intelligence Platform is an applied AI/ML project focused on:

* Payment failure prediction
* Cash forecasting
* Payment anomaly detection
* Operations intelligence
* Future GenAI/RAG support for payment operations

## Current Status

### Week 1: Data Foundation — Completed

Week 1 completed the data foundation for the Payments Intelligence Platform:

* Synthetic payment data generation
* Raw data validation
* Data cleaning
* Feature engineering
* Exploratory failure-pattern analysis
* End-to-end pipeline orchestration
* Data quality tests with pytest

### Week 2: Payment Failure Classifier — Completed

Week 2 completed the first applied ML model lifecycle:

* Baseline payment failure classifier
* Model comparison across Logistic Regression and Random Forest
* Threshold tuning for operational triage
* Model signal and feature importance analysis
* Model evaluation report
* Saved model artifact
* Inference script for scoring new payment records

The current model is an experimental baseline. It is useful for learning and portfolio demonstration, but it is not yet production-ready.

## Next Step

The next project phase will focus on one of the following:

1. Improving the payment failure model with stronger features
2. Starting cash forecasting
3. Building an API layer for model serving

### Week 3: Cash Forecasting — Completed

Week 3 added a second applied AI use case to the Payments Intelligence Platform.

Completed work:

* Daily cash forecasting dataset
* Baseline forecasts using previous day, 3-day average, and 7-day average
* Time-based train/test split
* Linear Regression and Random Forest regression models
* Forecast evaluation using MAE, RMSE, and MAPE
* Cash forecasting model evaluation report
* Saved cash forecasting model artifact
* Inference script for scoring new daily cash activity records

The current best experimental forecasting model is a Random Forest Regressor. It improved MAE by 3.22% compared with the 7-day moving average baseline.

The model is useful for learning and portfolio demonstration, but it is not yet production-ready.

## Next Step

The next project phase will focus on one of the following:

1. Payment anomaly detection
2. Improving the payment failure model with stronger historical features
3. Building an API layer for model serving

