# Payments Intelligence Platform

## Project Overview

The Payments Intelligence Platform is an applied AI/ML engineering project focused on payment operations, payment failure prediction, cash forecasting, anomaly detection, and future GenAI-powered operations support.

This project is designed as a hands-on portfolio project to demonstrate how AI and machine learning can be applied to enterprise payment workflows.

The current version focuses on building a clean, validated, ML-ready payment transaction dataset.

## Business Problem

Payment platforms process large volumes of transactions across different currencies, countries, payment types, and channels. Failed payments, delayed settlements, file delivery issues, and operational exceptions can create business risk, manual investigation effort, liquidity impact, and customer-service friction.

This project explores how AI/ML techniques can support payment operations by enabling:

* Payment failure prediction
* Cash-flow forecasting
* Payment anomaly detection
* Operational risk scoring
* Future GenAI/RAG-based runbook assistance

## Current Scope: Week 1 Data Foundation

Week 1 focuses on creating the foundation required for future machine learning work.

The current pipeline performs:

1. Synthetic payment data generation
2. Raw data validation
3. Data cleaning
4. Clean data validation
5. Feature engineering
6. Final feature dataset validation
7. Data quality testing with pytest
8. Exploratory failure-pattern analysis

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

  src/
    __init__.py
    data/
      __init__.py
      generate_payments.py
      validate_payments.py
      clean_payments.py
      feature_engineering.py
      payment_failure_eda.py
      run_pipeline.py

  tests/
    test_data_quality.py
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
17 passed
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

### Recommended Next Improvements

Future improvements should include:

1. Tune Isolation Forest contamination rate.
2. Add counterparty-level historical behavior features.
3. Add rolling amount and volume features by counterparty.
4. Add currency-pair and country-pair anomaly features.
5. Add alert suppression rules.
6. Add analyst feedback labels.
7. Track anomaly investigation outcomes.
8. Add monitoring and drift detection.
9. Serve anomaly scoring through an API.