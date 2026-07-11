# Payment Failure Classifier v2 Evaluation

## Executive Summary

The Payment Failure Classifier v2 improves the original Week 2 baseline by adding stronger failure-risk features and tuning the operating threshold.

The goal of v2 was not just to increase accuracy. The main business goal was to reduce false positives and alert volume while keeping useful recall for failed payments.

The best v2 setup is:

```text
Model: Logistic Regression with balanced class weights
Threshold: 0.55
Dataset: payments_failure_features_v2.csv
Model artifact: models/payment_failure_classifier_v2.pkl
```

## Business Problem

Payment failures create operational risk, manual investigation effort, settlement delays, and customer-service friction.

A useful payment failure model should help operations teams identify payments that may need review before release.

However, the model must balance two risks:

```text
False positive: normal payment incorrectly sent to review
False negative: failed payment missed by the model
```

The Week 2 baseline caught some failures, but generated too many false positives.

## Week 2 Baseline

The Week 2 baseline used Logistic Regression with balanced class weights.

Baseline performance:

| Metric          |  Value |
| --------------- | -----: |
| Precision       | 0.0779 |
| Recall          | 0.5357 |
| F1              | 0.1361 |
| False positives |    355 |
| False negatives |     26 |
| Alert rate      |  38.5% |

## Baseline Interpretation

The baseline model had useful recall, but low precision.

It caught:

```text
30 out of 56 failed payments
```

But it also incorrectly alerted:

```text
355 normal payments
```

This would create too much manual review work for an operations team.

## V2 Feature Improvements

The v2 dataset added stronger business and historical-risk features.

Input dataset:
```text
data/processed/payments_failure_features_v2.csv
```
V2 dataset shape:
```text
Rows: 5000
Columns: 39
Model features before encoding: 36
```
New v2 feature groups included:

## Historical Failure-Rate Features

```text
payment_type_failure_rate
channel_failure_rate
country_failure_rate
currency_failure_rate
payment_type_channel_failure_rate
```

These features help the model learn which payment types, channels, countries, and currencies have higher historical failure risk.

## Interaction Features

```text
risk_payment_interaction
amount_risk_interaction
prior_failure_risk_interaction
log_risk_adjusted_amount
```
These features capture relationships between amount, risk score, and prior failure history.

## Business-Risk Boolean Features

```text
high_value_high_risk
swift_file_payment
international_high_risk
large_prior_failure_payment
```

These features represent known payment-risk patterns.

## Leakage Note

The v2 dataset includes dataset-level failure-rate features for portfolio experimentation.

In a real production system, these features must be computed only from historical data available before prediction time.

For example, a production-safe ```text payment_type_failure_rate``` should use only prior payments, not future outcomes.

## V2 Model Candidates

The following v2 models were evaluated:

| Model                           | Precision | Recall |     F1 | False Positives | False Negatives | Alert Rate |
| ------------------------------- | --------: | -----: | -----: | --------------: | --------------: | ---------: |
| Logistic Regression Balanced v2 |    0.0929 | 0.6071 | 0.1611 |             332 |              22 |      36.6% |
| Random Forest Balanced v2       |    0.1429 | 0.0893 | 0.1099 |              30 |              51 |       3.5% |
| Gradient Boosting v2            |    0.0000 | 0.0000 | 0.0000 |               4 |              56 |       0.4% |


## Model Candidate Interpretation

### Logistic Regression Balanced v2
This was the best model by F1 score.

It improved precision, recall, and F1 compared with the Week 2 baseline.

However, at the default threshold, the alert rate was still high.

### Random Forest Balanced v2
Random Forest significantly reduced false positives, but recall dropped too much.

It missed most failed payments, making it less useful for payment failure triage.

### Gradient Boosting v2
Gradient Boosting produced almost no useful failure detection in this setup.

It avoided alerts but missed all failed payments.

### Threshold Tuning
The best v2 model was Logistic Regression Balanced v2.

Threshold tuning was performed to find a better operating point.

The selected threshold was:
```text
0.55
```

This threshold was selected because it met the practical target:
```text
Recall >= 40%
Alert rate <= 30%
```

## Recommended V2 Operating Result

| Metric          | V2 at Threshold 0.55 |
| --------------- | -------------------: |
| Precision       |               0.1049 |
| Recall          |               0.5000 |
| F1              |               0.1734 |
| False positives |                  239 |
| False negatives |                   28 |
| True positives  |                   28 |
| Alert rate      |                26.7% |

## Comparison vs Week 2 Baseline

| Metric          | Week 2 Baseline | V2 Threshold 0.55 |                  Change |
| --------------- | --------------: | ----------------: | ----------------------: |
| Precision       |          0.0779 |            0.1049 |                 +0.0270 |
| Recall          |          0.5357 |            0.5000 |                 -0.0357 |
| F1              |          0.1361 |            0.1734 |                 +0.0373 |
| False positives |             355 |               239 |                    -116 |
| False negatives |              26 |                28 |                      +2 |
| Alert rate      |           38.5% |             26.7% | -11.8 percentage points |

## Business Interpretation

The v2 model creates a more realistic review queue.

Compared with the Week 2 baseline, it reduces false positives by:

```text
116 payments
```
while only increasing false negatives by:

```text
2 payments
```
This is a better operational tradeoff because it lowers review workload while preserving useful failure detection.

## Saved V2 Model Artifact
The v2 model was saved to:

```text
models/payment_failure_classifier_v2.pkl
```
The artifact includes:

```text
trained model pipeline
feature columns
numeric/categorical/boolean feature groups
target column
model version
recommended threshold
training dataset path
baseline comparison notes
```
## V2 Inference

The v2 model can score new payments using:

```text
python -m src.models.predict_failure_v2
```
The inference script returns:

```text
predicted failure flag
predicted failure probability
risk band
recommended action
```
Example actions:

```text
Allow normal processing
Monitor or queue for secondary review
Review before release
```

## Current Model Status
Current status:

```text
Improved experimental payment failure classifier
```

Not yet:

```text
Production-ready payment failure triage system
```
## Limitations
The current v2 model has several limitations:

It is trained on synthetic data.
Historical failure-rate features are dataset-level approximations.
Production-safe historical features need time-aware calculation.
The model has not been calibrated.
The model has not been tested on real payment outcomes.
The model does not yet log predictions for monitoring.
The model does not yet include drift detection.
The model has not been deployed beyond local inference/API workflows.

## Recommended Next Improvements
Future improvements should include:

1. Time-aware historical feature engineering.
2. Counterparty-level rolling failure rates.
3. Payment-type/channel rolling failure rates.
4. Probability calibration.
5. Prediction logging.
6. Model monitoring and drift detection.
7. Human-in-the-loop feedback capture.
8. API upgrade to serve the v2 model.
9. Model governance documentation.