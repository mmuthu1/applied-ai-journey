# Payment Failure Model Evaluation Report

## 1. Executive Summary

This report evaluates the first machine learning models built for the Payments Intelligence Platform.

The objective is to predict whether a payment is likely to fail using pre-outcome payment attributes such as amount, currency, country, payment type, channel, settlement window, counterparty risk score, and engineered risk features.

The current model is useful as an experimental baseline, but it is not yet production-ready for operational triage.

Key findings:

* The dataset is imbalanced, with a payment failure rate of approximately 5.62%.
* Accuracy is misleading because a no-skill model can achieve 94.4% accuracy by predicting every payment as not failed.
* Logistic Regression with balanced class weights catches more failed payments than the no-skill model and regular Logistic Regression.
* The balanced Logistic Regression model improves recall but creates too many false positives.
* Threshold tuning improves the operational tradeoff but does not yet produce a practical triage queue.
* Further feature engineering and model improvement are needed before production use.

## 2. Business Problem

Payment failures can create operational risk, liquidity impact, manual investigation effort, client-service issues, and delayed settlement.

The business goal is to identify payments that are more likely to fail before the failure occurs, so operations teams can prioritize review and intervention.

A useful model should help answer:

```text
Which payments should operations review first?
```

However, the model must balance two competing risks:

| Risk           | Meaning                                              |
| -------------- | ---------------------------------------------------- |
| False positive | A normal payment is incorrectly flagged for review   |
| False negative | A payment that actually fails is missed by the model |

False negatives can be costly because they represent missed failed payments. False positives also matter because they create additional manual-review workload.

## 3. Dataset Summary

The model uses the engineered dataset:

```text
data/processed/payments_features.csv
```

Dataset size:

```text
Total records: 5,000
Failed payments: 281
Failure rate: 5.62%
```

The target column is:

```text
is_failed
```

The feature dataset excludes leakage columns:

```text
status
failure_reason
```

These columns were excluded because they reveal the payment outcome and would make the model artificially strong.

## 4. Feature Groups

The model uses the following feature groups.

### Numeric Features

* `amount`
* `counterparty_risk_score`
* `historical_failure_count`
* `payment_year`
* `payment_month`
* `payment_day`
* `day_of_week`
* `hour_of_day`
* `amount_log`
* `settlement_speed_days`
* `risk_adjusted_amount`

### Categorical Features

* `currency`
* `country`
* `payment_type`
* `channel`
* `settlement_window`
* `amount_bucket`
* `risk_band`
* `prior_failure_band`

### Boolean Features

* `is_weekend`
* `is_high_value`
* `has_prior_failures`
* `is_large_international`

## 5. Why Accuracy Is Misleading

The test set contains 1,000 payments:

```text
944 not failed
56 failed
```

A no-skill model that predicts every payment as not failed achieves:

```text
Accuracy: 94.4%
Recall: 0.0%
F1 score: 0.0%
```

This model appears strong by accuracy, but it catches zero failed payments.

Therefore, accuracy alone is not a useful success metric for this problem.

More relevant metrics are:

| Metric          | Why it matters                                             |
| --------------- | ---------------------------------------------------------- |
| Precision       | Measures how many flagged payments actually fail           |
| Recall          | Measures how many actual failed payments the model catches |
| F1 score        | Balances precision and recall                              |
| False positives | Measures operational review burden                         |
| False negatives | Measures missed failed-payment risk                        |

## 6. Baseline Logistic Regression Model

The first useful baseline was Logistic Regression with balanced class weights.

Model setup:

```text
Model: Logistic Regression
Preprocessing: StandardScaler for numeric features, OneHotEncoder for categorical features
Class imbalance handling: class_weight='balanced'
Train/test split: 80/20 with stratification
```

Results:

```text
Accuracy:  0.6190
Precision: 0.0779
Recall:    0.5357
F1 score:  0.1361
```

Confusion matrix:

```text
[[589 355]
 [ 26  30]]
```

Business interpretation:

| Result          | Count | Meaning                               |
| --------------- | ----: | ------------------------------------- |
| True negatives  |   589 | Normal payments correctly ignored     |
| False positives |   355 | Normal payments unnecessarily flagged |
| False negatives |    26 | Failed payments missed                |
| True positives  |    30 | Failed payments correctly identified  |

The model catches 30 out of 56 failed payments, but it also creates 355 false positives.

This model is better than the no-skill baseline because it catches actual failed payments, but it is too noisy for practical operational triage.

## 7. Model Comparison

The following models were compared:

1. No-skill baseline
2. Logistic Regression
3. Logistic Regression with balanced class weights
4. Random Forest with balanced class weights

Summary:

| Model                        | Accuracy | Precision | Recall |    F1 | False Positives | False Negatives | True Positives |
| ---------------------------- | -------: | --------: | -----: | ----: | --------------: | --------------: | -------------: |
| No-skill baseline            |    0.944 |     0.000 |  0.000 | 0.000 |               0 |              56 |              0 |
| Logistic Regression          |    0.944 |     0.000 |  0.000 | 0.000 |               0 |              56 |              0 |
| Logistic Regression Balanced |    0.619 |     0.078 |  0.536 | 0.136 |             355 |              26 |             30 |
| Random Forest Balanced       |    0.930 |     0.150 |  0.054 | 0.079 |              17 |              53 |              3 |

Interpretation:

* The no-skill baseline and regular Logistic Regression achieve high accuracy but catch no failed payments.
* Logistic Regression Balanced has the best recall and F1 score.
* Random Forest Balanced has fewer false positives but misses most failed payments.
* Logistic Regression Balanced is the best current model for finding failures, but it is operationally noisy.

## 8. Threshold Tuning

The balanced Logistic Regression model was evaluated at multiple probability thresholds.

The default threshold is:

```text
0.50
```

At threshold 0.50:

```text
Recall: 53.57%
Precision: 7.79%
False positives: 355
False negatives: 26
Alert rate: 38.5%
```

The best F1 threshold was:

```text
Threshold: 0.60
F1 score: 0.1511
Recall: 30.36%
Precision: 10.06%
False positives: 152
False negatives: 39
Alert rate: 16.9%
```

Threshold 0.60 reduces the operational review burden compared with threshold 0.50, but it also catches fewer failed payments.

A practical rule was tested:

```text
Recall >= 40%
Alert rate <= 30%
```

No threshold met this rule.

This means the current model is not yet strong enough for a practical triage queue.

## 9. Model Signal Analysis

Logistic Regression coefficients and Random Forest feature importance were reviewed to understand model behavior.

Signals that push toward failed-payment prediction include:

* `is_large_international`
* `payment_type_SWIFT`
* `channel_FILE`
* `counterparty_risk_score`
* `amount`

Random Forest feature importance also highlighted:

* `risk_adjusted_amount`
* `counterparty_risk_score`
* `amount`
* `amount_log`
* `payment_type_SWIFT`
* `historical_failure_count`
* `channel_FILE`

These signals broadly align with the EDA findings and business intuition.

However, some Logistic Regression coefficients were counterintuitive, including negative weights for some high-value features. This may be caused by correlated engineered features such as:

* `amount`
* `amount_log`
* `amount_bucket`
* `is_high_value`
* `is_large_international`
* `risk_adjusted_amount`

Because these features overlap, coefficient interpretation should be treated carefully.

## 10. Operational Recommendation

The current best experimental model is:

```text
Logistic Regression Balanced
```

The best experimental threshold is:

```text
0.60
```

Reason:

* It has the best F1 score among tested thresholds.
* It reduces alert volume compared with threshold 0.50.
* It provides a more manageable operational triage queue.

However, this should not be considered production-ready.

Recommended current positioning:

```text
Experimental baseline model for payment failure triage.
```

Not:

```text
Production-ready payment failure prediction system.
```

## 11. Limitations

Current limitations:

1. The dataset is synthetic.
2. The model has low precision.
3. The model creates too many false positives at useful recall levels.
4. No threshold meets the practical triage rule.
5. Feature importance shows some correlated-feature effects.
6. The model does not yet include counterparty-level history.
7. The model does not include rolling failure rates.
8. The model does not include real operational feedback loops.
9. The model is not calibrated for production probability interpretation.
10. No model monitoring or drift detection exists yet.

## 12. Recommended Next Improvements

Recommended next steps:

1. Add stronger counterparty-level features.
2. Create rolling historical failure-rate features by payment type, channel, country, and counterparty.
3. Add probability bands for operational triage:

   * High risk
   * Medium risk
   * Low risk
4. Tune model class weights and thresholds together.
5. Try additional models later, such as Gradient Boosting.
6. Create a model artifact and save it for later API serving.
7. Add a formal evaluation report generation script.
8. Track model versions and evaluation metrics.
9. Add monitoring and drift checks in later MLOps phases.

## 13. Conclusion

The first payment failure classifier successfully demonstrates the ML workflow:

```text
→ feature dataset
→ train/test split
→ preprocessing
→ model training
→ model comparison
→ threshold tuning
→ business interpretation
```

The current model proves that failure prediction is possible, but it also shows that the first baseline is not yet strong enough for production triage.

The most important learning is that applied AI model evaluation must consider business tradeoffs, not just technical metrics.

For this use case, the right question is not:

```text
Which model has the highest accuracy?
```

The right question is:

```text
Which model catches enough failures without overwhelming operations with false alerts?
```

This project will continue by improving features, tuning models, and preparing the model for future API serving and production-style deployment.
