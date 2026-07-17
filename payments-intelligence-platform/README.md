# Payments Intelligence Platform

## Project Overview

The Payments Intelligence Platform is an applied AI/ML engineering project focused on payment operations, payment failure prediction, cash forecasting, anomaly detection, and future GenAI-powered operations support.

This project is designed as a hands-on portfolio project to demonstrate how AI and machine learning can be applied to enterprise payment workflows.

The current version includes data validation, feature engineering, machine learning models, saved model artifacts, FastAPI model-serving endpoints, and automated data quality/API tests.

## Business Problem

Payment platforms process large volumes of transactions across different currencies, countries, payment types, and channels. Failed payments, delayed settlements, file delivery issues, and operational exceptions can create business risk, manual investigation effort, liquidity impact, and customer-service friction.

This project explores how AI/ML techniques can support payment operations by enabling:

* Payment failure prediction
* Cash-flow forecasting
* Payment anomaly detection
* Operational risk scoring
* Future GenAI/RAG-based runbook assistance

## Current Platform Capabilities

The current platform includes:

1. Synthetic payment data generation
2. Raw and clean data validation
3. Feature engineering
4. Payment failure prediction
5. Cash forecasting
6. Payment anomaly detection
7. Saved model artifacts
8. FastAPI model-serving endpoints
9. Data quality testing with pytest
10. API endpoint testing with pytest
11. Exploratory failure-pattern and anomaly analysis
12. Model metadata registry
13. Prediction audit logging
14. Data drift monitoring
15. Generated model monitoring report
16. Payment operations runbooks
17. Local GenAI/RAG-style document store
18. Local document retrieval logic
19. GenAI-style operations assistant
20. Assistant retrieval evaluation

## Project Structure

```text
payments-intelligence-platform/
  README.md
  requirements.txt
  pytest.ini

  data/
    raw/
      payments_raw.csv
    processed/
      payments_clean.csv
      payments_features.csv
      cash_forecast_daily.csv
      payment_anomaly_features.csv
      payment_anomaly_scored.csv
      payments_failure_features_v2.csv
    monitoring/
      model_metadata.json
      prediction_logs.csv
      data_drift_summary.csv

  models/
    payment_failure_classifier.pkl
    payment_failure_classifier_v2.pkl
    cash_forecast_model.pkl
    payment_anomaly_detector.pkl

  reports/
    payment_failure_model_evaluation.md
    cash_forecasting_model_evaluation.md
    payment_anomaly_detection_report.md
    payment_failure_model_v2_evaluation.md
    model_monitoring_report.md

  src/
    data/
      generate_payments.py
      validate_payments.py
      clean_payments.py
      feature_engineering.py
      payment_failure_eda.py
      run_pipeline.py

    models/
      train_failure_classifier.py
      compare_failure_classifiers.py
      tune_failure_threshold.py
      analyze_model_signals.py
      save_failure_model.py
      predict_failure.py
      diagnose_failure_model.py
      build_failure_features_v2.py
      train_failure_classifier_v2.py
      tune_failure_threshold_v2.py
      save_failure_model_v2.py
      predict_failure_v2.py

    forecasting/
      prepare_cash_forecast_data.py
      baseline_cash_forecast.py
      train_cash_forecast_model.py
      save_cash_forecast_model.py
      predict_cash_forecast.py

    anomaly/
      prepare_anomaly_dataset.py
      train_isolation_forest.py
      analyze_anomaly_results.py
      save_anomaly_model.py
      predict_anomaly.py

    api/
      main.py
      schemas.py
      model_loader.py

    monitoring/
      model_metadata.py
      prediction_logger.py
      data_drift_check.py
      generate_monitoring_report.py

  tests/
    test_data_quality.py
    test_api.py
```

## Dataset Schema

The raw synthetic payment dataset includes the following columns:

| Column                     | Description                                              |
| -------------------------- | -------------------------------------------------------- |
| `payment_id`               | Unique payment identifier                                |
| `payment_date`             | Payment timestamp                                        |
| `amount`                   | Payment amount                                           |
| `currency`                 | Payment currency                                         |
| `country`                  | Payment country                                          |
| `payment_type`             | Payment type such as ACH, WIRE, SWIFT, RTP               |
| `channel`                  | Payment channel such as API, FILE, BATCH, ONLINE, MOBILE |
| `counterparty_risk_score`  | Simulated risk score between 0 and 1                     |
| `historical_failure_count` | Prior simulated failure count                            |
| `settlement_window`        | Settlement timing category                               |
| `status`                   | Payment status                                           |
| `failure_reason`           | Failure reason for failed payments                       |
| `is_failed`                | Target variable for future ML model                      |

## ML Target

The primary target variable for the Week 2 model is:

```text
is_failed
```

This target will be used to train a payment failure prediction model.

## Leakage Prevention

The following columns are removed from the ML feature dataset:

```text
status
failure_reason
```

These columns are excluded because they reveal the payment outcome and would cause data leakage during model training.

## Engineered Features

The feature engineering step creates ML-ready features such as:

| Feature                  | Purpose                                      |
| ------------------------ | -------------------------------------------- |
| `payment_year`           | Year extracted from payment date             |
| `payment_month`          | Month extracted from payment date            |
| `payment_day`            | Day extracted from payment date              |
| `day_of_week`            | Day-of-week feature                          |
| `hour_of_day`            | Hour-of-day feature                          |
| `is_weekend`             | Weekend flag                                 |
| `amount_log`             | Log-transformed amount                       |
| `is_high_value`          | High-value payment flag                      |
| `amount_bucket`          | Amount category                              |
| `risk_band`              | Risk score category                          |
| `has_prior_failures`     | Prior failure flag                           |
| `prior_failure_band`     | Prior failure count category                 |
| `settlement_speed_days`  | Numeric settlement speed                     |
| `is_large_international` | Large non-US payment flag                    |
| `risk_adjusted_amount`   | Amount multiplied by counterparty risk score |

## How to Run the Pipeline

From the `payments-intelligence-platform` folder:

```bash
python -m src.data.run_pipeline
```

This command runs the full data pipeline:

```text
→ generate raw payments
→ validate raw payments
→ clean payments
→ validate clean payments
→ engineer ML features
→ validate final feature dataset
```

Expected final output:

```text
Pipeline completed successfully
```

## How to Run Tests

From the `payments-intelligence-platform` folder:

```bash
pytest
```

The current test suite validates:

* Raw data file exists
* Clean data file exists
* Feature dataset exists
* Feature dataset has records
* Expected record count is 5,000
* Payment IDs are not missing
* Payment IDs are unique
* Amount values are positive
* `is_failed` column exists
* Failure rate is within a reasonable range
* Currency values are valid
* Payment type values are valid
* Channel values are valid
* Risk scores are between 0 and 1
* Leakage columns are removed
* Required engineered features exist

Current result:

```text
22 passed
```

## Exploratory Data Analysis Findings

The EDA shows that payment failures are a minority class:

```text
Overall failure rate: 5.62%
Failed payments: 281 out of 5,000
```

This means model evaluation should not rely on accuracy alone. Week 2 model evaluation should include:

* Precision
* Recall
* F1 score
* Confusion matrix

Key observed patterns:

| Area          | Finding                                                  |
| ------------- | -------------------------------------------------------- |
| Payment type  | SWIFT has the highest failure rate                       |
| Channel       | FILE has the highest failure rate                        |
| Amount        | Very high-value payments have higher failure rates       |
| Risk          | Failed payments show higher average risk-adjusted amount |
| Class balance | Failed payments are a minority class                     |

## Week 2 Roadmap

## Week 2: Payment Failure Classifier

Week 2 focused on building the first machine learning model for the Payments Intelligence Platform.

The goal was to predict whether a payment is likely to fail using pre-outcome payment attributes and engineered features.

### Week 2 Deliverables

Week 2 added the following model development components:

| Component                      | File                                          |
| ------------------------------ | --------------------------------------------- |
| Baseline classifier            | `src/models/train_failure_classifier.py`      |
| Model comparison               | `src/models/compare_failure_classifiers.py`   |
| Threshold tuning               | `src/models/tune_failure_threshold.py`        |
| Model signal analysis          | `src/models/analyze_model_signals.py`         |
| Model evaluation report        | `reports/payment_failure_model_evaluation.md` |
| Saved model artifact           | `models/payment_failure_classifier.pkl`       |
| Model artifact creation script | `src/models/save_failure_model.py`            |
| Inference script               | `src/models/predict_failure.py`               |

### Modeling Approach

The classifier uses the engineered feature dataset:

```text
data/processed/payments_features.csv
```

Target column:

```text
is_failed
```

The model excludes leakage columns such as:

```text
status
failure_reason
```

because those columns reveal the final payment outcome.

### Models Evaluated

The following models were evaluated:

1. No-skill baseline
2. Logistic Regression
3. Logistic Regression with balanced class weights
4. Random Forest with balanced class weights

The best current experimental model is:

```text
Logistic Regression with class_weight='balanced'
```

This model is useful as an experimental baseline because it catches more failed payments than the no-skill baseline, but it is not production-ready because precision is still low and false positives remain high.

### Key Model Findings

The dataset is imbalanced, with a payment failure rate of approximately 5.62%.

A no-skill model can achieve high accuracy by predicting every payment as not failed, but it catches zero failed payments. Therefore, accuracy alone is not a useful metric for this use case.

More relevant metrics include:

* Precision
* Recall
* F1 score
* False positives
* False negatives
* Alert volume
* Operational cost

### Threshold Tuning

Threshold tuning showed that different probability cutoffs create different operational tradeoffs.

A threshold of `0.60` produced the best F1 score among tested thresholds and reduced alert volume compared with the default `0.50` threshold.

However, no threshold met the practical triage rule:

```text
Recall >= 40%
Alert rate <= 30%
```

This means the current model is not yet strong enough for a practical production triage queue.

### Model Signal Analysis

Model signal analysis showed that several features align with business intuition:

* `payment_type_SWIFT`
* `channel_FILE`
* `counterparty_risk_score`
* `risk_adjusted_amount`
* `amount`
* `amount_log`
* `historical_failure_count`

Some Logistic Regression coefficients were counterintuitive because several engineered features overlap, such as amount, amount bucket, high-value flag, and risk-adjusted amount. This is a useful reminder that model interpretability requires care when features are correlated.

### Inference

The saved model can score new payment records using:

```bash
python -m src.models.predict_failure
```

The inference script outputs:

* Predicted failure class
* Predicted failure probability
* Risk band
* Recommended action

Example recommended actions include:

* `Allow normal processing`
* `Monitor or queue for secondary review`
* `Review before release`

### How to Run Week 2 Model Scripts

From the `payments-intelligence-platform` folder:

```bash
python -m src.models.train_failure_classifier
python -m src.models.compare_failure_classifiers
python -m src.models.tune_failure_threshold
python -m src.models.analyze_model_signals
python -m src.models.save_failure_model
python -m src.models.predict_failure
```

### Current Model Status

Current status:

```text
Experimental baseline model
```

Not yet:

```text
Production-ready payment failure prediction system
```

### Recommended Next Improvements

Future improvements should include:

1. Stronger counterparty-level history features
2. Rolling failure-rate features
3. Probability bands for operational triage
4. Class-weight and threshold tuning together
5. Additional models such as Gradient Boosting
6. Model calibration
7. API serving with FastAPI
8. Monitoring and drift detection

## Week 3: Cash Forecasting

Week 3 added the second applied AI/ML use case to the Payments Intelligence Platform: next-day cash forecasting.

The goal was to forecast the next day’s total payment amount using daily payment activity, rolling averages, payment counts, failed payment rates, high-value payment rates, and calendar features.

### Week 3 Deliverables

| Component                                  | File                                            |
| ------------------------------------------ | ----------------------------------------------- |
| Daily cash forecasting dataset preparation | `src/forecasting/prepare_cash_forecast_data.py` |
| Baseline cash forecasts                    | `src/forecasting/baseline_cash_forecast.py`     |
| ML cash forecasting models                 | `src/forecasting/train_cash_forecast_model.py`  |
| Cash forecasting evaluation report         | `reports/cash_forecasting_model_evaluation.md`  |
| Saved cash forecast model artifact         | `models/cash_forecast_model.pkl`                |
| Model artifact creation script             | `src/forecasting/save_cash_forecast_model.py`   |
| Cash forecast inference script             | `src/forecasting/predict_cash_forecast.py`      |

### Forecasting Dataset

The forecasting dataset is created from cleaned payment transactions:

```text
data/processed/payments_clean.csv
```

and saved as:

```text
data/processed/cash_forecast_daily.csv
```

The dataset converts transaction-level payments into daily aggregates.

Target column:

```text
next_day_total_amount
```

This represents the next day’s total payment amount.

### Forecasting Features

The forecasting dataset includes features such as:

* `daily_payment_count`
* `daily_total_amount`
* `daily_average_amount`
* `daily_median_amount`
* `failed_payment_count`
* `high_value_payment_count`
* `failed_payment_rate`
* `high_value_payment_rate`
* `previous_day_total_amount`
* `rolling_3_day_avg_amount`
* `rolling_7_day_avg_amount`
* `rolling_3_day_payment_count`
* `rolling_7_day_payment_count`
* `day_of_week`
* `month`
* `day_of_month`
* `is_weekend`

### Baseline Forecasts

Before training ML models, simple baseline forecasts were evaluated:

1. Previous day amount
2. 3-day moving average
3. 7-day moving average

The best baseline was:

```text
7-day moving average forecast
```

Baseline performance:

```text
MAE:  $923,898.74
RMSE: $1,204,937.67
MAPE: 32.27%
```

This baseline became the benchmark that ML models needed to beat.

### ML Forecasting Models

The following models were evaluated using a time-based train/test split:

1. 7-day moving average baseline
2. Linear Regression
3. Random Forest Regressor

A time-based split was used because forecasting should train on earlier dates and test on later dates.

Training period:

```text
2024-01-07 to 2024-05-23
```

Test period:

```text
2024-05-24 to 2024-06-27
```

### Model Results

| Model                         |         MAE |          RMSE |   MAPE | Improvement vs Baseline |
| ----------------------------- | ----------: | ------------: | -----: | ----------------------: |
| Random Forest Regressor       | $894,179.20 | $1,163,787.60 | 30.00% |                  +3.22% |
| 7-day moving average baseline | $923,898.74 | $1,204,937.67 | 32.27% |                Baseline |
| Linear Regression             | $926,054.91 | $1,134,513.00 | 31.59% |                  -0.23% |

The current best experimental forecasting model is:

```text
Random Forest Regressor
```

It improved MAE by:

```text
$29,719.55
```

or:

```text
3.22%
```

compared with the 7-day moving average baseline.

### Cash Forecast Model Artifact

The trained cash forecasting model can be saved using:

```bash
python -m src.forecasting.save_cash_forecast_model
```

This creates:

```text
models/cash_forecast_model.pkl
```

The saved artifact includes:

* Trained Random Forest Regressor
* Feature column list
* Target column name

### Cash Forecast Inference

The saved model can score new daily cash activity records using:

```bash
python -m src.forecasting.predict_cash_forecast
```

The inference script outputs:

* Predicted next-day total amount
* Forecast band: LOW, MEDIUM, or HIGH
* Forecast vs 7-day average
* Forecast variance percentage
* Recommended action

Example recommendation categories:

* `Normal cash operations planning`
* `Monitor expected payment activity and exception volume`
* `Review liquidity, staffing, and high-value payment queue`

### Current Forecasting Model Status

Current status:

```text
Experimental cash forecasting model
```

Not yet:

```text
Production-ready liquidity forecasting system
```

The workflow is correct, but the model was trained on synthetic data and only 173 daily records. More historical data and richer calendar/business features are needed before production use.

### Recommended Next Improvements

Future improvements should include:

1. More historical daily records
2. Holiday and banking calendar features
3. Month-end and quarter-end indicators
4. Currency-specific cash forecasts
5. Inflow and outflow separation
6. Payment-type-specific aggregates
7. Prediction intervals or confidence bands
8. Hyperparameter tuning
9. Model monitoring and drift detection

## Week 4: Payment Anomaly Detection

Week 4 added the third applied AI/ML use case to the Payments Intelligence Platform: payment anomaly detection.

The goal was to identify unusual payment transactions using a combination of business-rule anomaly logic and unsupervised machine learning.

### Week 4 Deliverables

| Component                        | File                                          |
| -------------------------------- | --------------------------------------------- |
| Payment anomaly feature dataset  | `src/anomaly/prepare_anomaly_dataset.py`      |
| Isolation Forest anomaly model   | `src/anomaly/train_isolation_forest.py`       |
| Anomaly results analysis         | `src/anomaly/analyze_anomaly_results.py`      |
| Payment anomaly detection report | `reports/payment_anomaly_detection_report.md` |
| Saved anomaly model artifact     | `models/payment_anomaly_detector.pkl`         |
| Model artifact creation script   | `src/anomaly/save_anomaly_model.py`           |
| Anomaly inference script         | `src/anomaly/predict_anomaly.py`              |

### Anomaly Dataset

The anomaly dataset is created from:

```text
data/processed/payments_features.csv
```

and saved as:

```text
data/processed/payment_anomaly_features.csv
```

The scored anomaly dataset is saved as:

```text
data/processed/payment_anomaly_scored.csv
```

### Rule-Based Anomaly Features

The rule-based layer flags known business-risk patterns, including:

* Extreme payment amount
* Top 1% payment amount
* Extreme risk-adjusted amount
* Top 1% counterparty risk score
* High-risk large payment
* Large FILE-channel payment
* High-risk SWIFT payment
* High-value payment with prior failures

Each payment receives:

```text
rule_anomaly_score
is_rule_based_anomaly
anomaly_reasons
```

### Isolation Forest Model

The unsupervised anomaly model uses:

```text
Isolation Forest
```

The model was configured with:

```text
contamination = 0.04
```

This asks the model to flag roughly 4% of records as anomalies, which is close to the rule-based anomaly rate.

### Rule vs Model Comparison

The anomaly workflow compares:

```text
business-rule anomalies
vs
Isolation Forest anomalies
```

Results:

| Category         | Count | Meaning                         |
| ---------------- | ----: | ------------------------------- |
| `NORMAL`         | 4,758 | No anomaly review needed        |
| `RULE_AND_MODEL` |   141 | Flagged by both rules and model |
| `MODEL_ONLY`     |    59 | Flagged only by model           |
| `RULE_ONLY`      |    42 | Flagged only by rules           |

Combined anomaly review workload:

```text
242 payments
4.84% of total payments
```

### Review Priority Queues

The anomaly workflow assigns review priorities:

| Priority         | Source           | Recommended Action                 |
| ---------------- | ---------------- | ---------------------------------- |
| `P1_HIGH`        | `RULE_AND_MODEL` | High-priority anomaly review       |
| `P2_INVESTIGATE` | `MODEL_ONLY`     | Investigate unusual pattern        |
| `P3_KNOWN_RISK`  | `RULE_ONLY`      | Review known business-risk pattern |
| `P4_NORMAL`      | `NORMAL`         | No anomaly review needed           |

### Failure Rate by Anomaly Source

Anomaly groups showed higher failure rates than normal payments:

| Source           | Failure Rate |
| ---------------- | -----------: |
| `RULE_ONLY`      |       11.90% |
| `MODEL_ONLY`     |       11.86% |
| `RULE_AND_MODEL` |        8.51% |
| `NORMAL`         |        5.40% |

This suggests the anomaly workflow is identifying groups with elevated payment failure risk.

### Anomaly Model Artifact

The trained anomaly detector can be saved using:

```bash
python -m src.anomaly.save_anomaly_model
```

This creates:

```text
models/payment_anomaly_detector.pkl
```

The saved artifact includes:

* Full sklearn Pipeline
* StandardScaler for numeric features
* OneHotEncoder for categorical features
* Boolean passthrough features
* Isolation Forest model
* Feature column list
* Contamination setting

### Anomaly Inference

The saved anomaly detector can score new payment records using:

```bash
python -m src.anomaly.predict_anomaly
```

The inference script outputs:

* Anomaly score
* Anomaly band
* Rule-based anomaly flag
* Model anomaly flag
* Anomaly source
* Review priority
* Recommended action
* Anomaly reasons

### Current Anomaly Model Status

Current status:

```text
Experimental anomaly detection workflow
```

Not yet:

```text
Production-ready payment anomaly monitoring system
```

The workflow is useful for learning, portfolio demonstration, and early anomaly triage design. It combines business rules with unsupervised model-based pattern discovery.

## Week 5: FastAPI Model Serving

The Week 5 milestone turns the Payments Intelligence Platform from standalone ML scripts into an API-backed application.

The FastAPI app serves three model workflows:

```text
GET  /
GET  /health
GET  /models

POST /predict/payment-failure
POST /predict/cash-forecast
POST /predict/payment-anomaly
```

### Payment Failure Prediction Endpoint

```text
POST /predict/payment-failure

Accepts payment-level details and returns:

predicted failure flag
predicted failure probability
risk band
recommended operational action

```

### Cash Forecast Prediction Endpoint

```text
POST /predict/cash-forecast

Accepts daily payment activity and returns:

predicted next-day total payment amount
forecast band
recommended action
difference from 7-day rolling average
percentage difference from 7-day rolling average

```

### Payment Anomaly Detection Endpoint

```text
POST /predict/payment-anomaly

Accepts payment-level details and returns:

model anomaly flag
rule-based anomaly flag
anomaly score
anomaly band
anomaly source
review priority
anomaly reasons
recommended action

```

### API Metadata Endpoint

```text
GET /models

Lists the available model-serving endpoints:

payment failure classifier
cash forecast model
payment anomaly detector

```

Running the API Locally:

```bash
uvicorn src.api.main:app --reload

Then open:

http://127.0.0.1:8000/docs

```

### API Tests

The API layer includes automated tests for:

- health check endpoint
- model metadata endpoint
- payment failure prediction endpoint
- cash forecast prediction endpoint
- payment anomaly detection endpoint

Run API tests:

The API layer includes automated tests for:

health check endpoint
model metadata endpoint
payment failure prediction endpoint
cash forecast prediction endpoint
payment anomaly detection endpoint

```bash
pytest tests/test_api.py -v
```

Run all tests:

```bash
pytest -v

Latest full test result:

22 passed

```

### Recommended Next Improvements

Future improvements should include:

1. Add request/response examples to the README.
2. Add stronger input validation for allowed categorical values.
3. Add API error handling.
4. Add prediction logging.
5. Add model/version metadata.
6. Add monitoring and drift checks.
7. Containerize the API with Docker.

## Week 6: Payment Failure Classifier v2

Week 6 improved the original payment failure classifier by adding stronger failure-risk features, comparing v2 model candidates, tuning the operating threshold, and saving an improved v2 model artifact.

### Week 6 Deliverables

| Component | File |
|---|---|
| Baseline diagnosis | `src/models/diagnose_failure_model.py` |
| V2 feature engineering | `src/models/build_failure_features_v2.py` |
| V2 model training | `src/models/train_failure_classifier_v2.py` |
| V2 threshold tuning | `src/models/tune_failure_threshold_v2.py` |
| V2 saved model artifact | `models/payment_failure_classifier_v2.pkl` |
| V2 model save script | `src/models/save_failure_model_v2.py` |
| V2 inference script | `src/models/predict_failure_v2.py` |
| V2 evaluation report | `reports/payment_failure_model_v2_evaluation.md` |

### V2 Improvement Summary

The Week 2 baseline created too many false positives.

The v2 model improved the operating tradeoff by using stronger feature engineering and a tuned threshold.

Recommended v2 setup:

```text
Model: Logistic Regression Balanced v2
Threshold: 0.55
```

Comparison against Week 2 baseline:

| Metric          | Week 2 Baseline | V2 Threshold 0.55 |
| --------------- | --------------: | ----------------: |
| Precision       |          0.0779 |            0.1049 |
| Recall          |          0.5357 |            0.5000 |
| F1              |          0.1361 |            0.1734 |
| False positives |             355 |               239 |
| False negatives |              26 |                28 |
| Alert rate      |           38.5% |             26.7% |

Business interpretation:
```text
V2 reduced false positives by 116 while only increasing false negatives by 2.
```

Current status:
```text
Improved experimental payment failure classifier
```

Not yet:
```text
Production-ready payment failure triage system
```
## Week 7: Monitoring, Governance, and MLOps Foundations

Week 7 added a basic monitoring and governance layer to make the Payments Intelligence Platform more operationally realistic.

The goal was to move beyond model training and API serving by adding visibility into model metadata, prediction activity, drift monitoring, and production-readiness gaps.

### Week 7 Deliverables

| Component | File |
|---|---|
| Model metadata registry | `src/monitoring/model_metadata.py` |
| Prediction logging utility | `src/monitoring/prediction_logger.py` |
| Data drift monitoring check | `src/monitoring/data_drift_check.py` |
| Monitoring report generator | `src/monitoring/generate_monitoring_report.py` |
| Model metadata output | `data/monitoring/model_metadata.json` |
| Prediction audit log | `data/monitoring/prediction_logs.csv` |
| Drift summary output | `data/monitoring/data_drift_summary.csv` |
| Monitoring report | `reports/model_monitoring_report.md` |

### Monitoring Capabilities Added

The platform now includes:

- model metadata tracking
- model version tracking
- model artifact tracking
- prediction audit logging
- data drift monitoring
- generated monitoring report
- documented governance limitations
- documented production-readiness recommendations

### Model Metadata Registry

The model metadata registry is generated at:

```text
data/monitoring/model_metadata.json
```
It tracks:

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

This helps answer governance questions such as which model version is available, which artifact is used, and whether a model is currently connected to the API.

Prediction Audit Logging

Prediction events are logged to:

```text
data/monitoring/prediction_logs.csv
```

The log captures:

timestamp
model name
model version
model type
record ID
prediction output
prediction probability or score
prediction band
recommended action
prediction source

The v2 payment failure inference flow now writes prediction events to this log, creating a simple audit trail.

### Data Drift Monitoring

The drift check is generated at:

```text
data/monitoring/data_drift_summary.csv
```

It compares a reference dataset against a simulated new scoring dataset across:

amount
counterparty risk score
historical failure count
risk-adjusted amount
payment type mix
channel mix
country mix
currency mix

This demonstrates how the platform could monitor whether incoming payment patterns are changing compared with the original reference data.

### Monitoring Report
The generated monitoring report is available at:

```text
reports/model_monitoring_report.md
```

The report summarizes:

model registry status
prediction log activity
data drift results
governance notes
current limitations
production-readiness recommendations

### Current Monitoring Status

Current status:

```text
Experimental monitoring and governance foundation
```
Not yet:

```text
Production-ready MLOps monitoring system
```

### Current Limitations

The monitoring layer is still portfolio-focused.

Current limitations include:

Prediction logs are file-based, not database-backed.
Drift checks use simulated new scoring data.
No automated alerting exists yet.
No monitoring dashboard exists yet.
No authentication or authorization exists.
No model approval workflow exists yet.
No scheduled monitoring jobs exist yet.
No production data is used.

### Recommended Next Improvements

Future improvements should include:

Connect prediction logging to all API endpoints.
Store prediction logs in a database.
Schedule drift checks.
Add automated monitoring alerts.
Add a monitoring dashboard.
Add model owner and reviewer metadata.
Add human-review outcome capture.
Add deployment and rollback procedures.

## Week 8: GenAI/RAG Operations Assistant

Week 8 added a local GenAI/RAG-style operations assistant for the Payments Intelligence Platform.

The goal was to build an assistant that can answer payment operations questions using local project documentation, runbooks, and reports.

This implementation is portfolio-safe and does not use confidential data or external APIs.

### Week 8 Deliverables

| Component | File |
|---|---|
| Payment failure review runbook | `docs/runbooks/payment_failure_review.md` |
| Payment anomaly review runbook | `docs/runbooks/anomaly_review.md` |
| Cash forecast review runbook | `docs/runbooks/cash_forecast_review.md` |
| Model monitoring review runbook | `docs/runbooks/model_monitoring_review.md` |
| Local document store | `src/genai/document_store.py` |
| Retrieval logic | `src/genai/retrieve_context.py` |
| Operations assistant | `src/genai/operations_assistant.py` |
| Assistant evaluation | `src/genai/evaluate_assistant.py` |
| Document store output | `data/genai/document_store.json` |
| Assistant evaluation report | `reports/genai_assistant_evaluation.md` |

### Assistant Workflow

The assistant follows a simple RAG-style workflow:

```text
user question
→ classify question type
→ retrieve relevant runbook/report context
→ generate structured operational response
```

The assistant can answer questions related to:

high-risk payment review
payment anomaly investigation
cash forecast interpretation
model monitoring and data drift review
model version and prediction logging questions

### Local Document Store

The local document store is generated at:

```text
data/genai/document_store.json
```

It includes:

  payment operations runbooks
  model evaluation reports
  monitoring report
  project README documentation

The document store currently contains:

```text
10 documents
```

### Retrieval Logic

The retrieval layer searches the local document store and returns the most relevant documents for a user question.

The retrieval logic currently uses:

  keyword tokenization
  stopword removal
  document scoring
  document-type weighting
  runbook preference for operational questions
  small keyword boosts for known operational scenarios

This is a simple local retrieval approach and does not require embeddings or external APIs.

### Operations Assistant

The operations assistant is implemented in:

```text
src/genai/operations_assistant.py
```

It provides structured answers with:

  question
  recommended operational steps
  important limitation
  primary source
  retrieved context

Example questions:

```text
What should operations do for a high risk payment?
How should we review a payment anomaly?
What does high data drift mean?
How should we interpret a cash forecast?
```

### Assistant Evaluation
The assistant evaluation is implemented in:

```text
src/genai/evaluate_assistant.py
```

The evaluation checks whether the assistant:

1. Classifies the question into the expected operations category.
2. Retrieves the expected runbook as the top supporting document.

Current evaluation result:

```text
Total questions: 8
Passed questions: 8
Failed questions: 0
Overall pass rate: 100%
```

### Current Assistant Status

Current status:

```text
Local experimental GenAI/RAG-style operations assistant
```

Not yet:

```text
Production-ready GenAI copilot
```

## Current Limitations

The assistant is still experimental and portfolio-focused.

Current limitations include:

1. It uses keyword-based retrieval, not embeddings.
2. It uses rule-based answer templates, not live LLM generation.
3. It uses local project documentation only.
4. It does not access live payment records.
5. It does not support multi-turn conversations yet.
6. It does not include source citations inside generated answers yet.
7. It has not been tested with real operations users.
8. It does not include access controls, audit permissions, or production guardrails.

## Recommended Next Improvements

Future improvements should include:

1. Add chunk-level retrieval.
2. Add semantic embeddings.
3. Add LLM-based answer generation.
4. Add source citations in final assistant answers.
5. Add more evaluation questions.
6. Add user feedback capture.
7. Add guardrails for uncertain or unsupported answers.
8. Connect assistant responses to actual prediction outputs and monitoring logs.

