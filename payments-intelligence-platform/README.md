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

## Cash Forecasting Report

The Week 3 cash forecasting model evaluation is documented here:

```text
reports/cash_forecasting_model_evaluation.md