# Payment Anomaly Review Runbook

## Purpose

This runbook explains how to review unusual payments flagged by the anomaly detection workflow.

The anomaly workflow combines business-rule anomaly detection with an unsupervised Isolation Forest model.

## Model Used

Current anomaly model:

```text
payment_anomaly_detector
```

The workflow combines:

    rule-based anomaly flags
    model-based anomaly flags
    anomaly source classification
    review priority assignment

## Anomaly Sources

### RULE_AND_MODEL

Meaning:

```text
Flagged by both business rules and the machine learning model
```
Recommended priority:

```text
P1_HIGH
```
Recommended action:

```text
High-priority anomaly review
```

Operations guidance:

    Review immediately.
    Check anomaly reasons.
    Check amount and risk-adjusted amount.
    Check payment type and channel.
    Check counterparty risk score.
    Confirm whether the payment is high value, high risk, or unusual for the channel.
    Escalate if the payment appears suspicious or operationally risky.

### MODEL_ONLY

Meaning:

```text
Flagged by the Isolation Forest model but not by business rules
```
Recommended priority:

```text
P2_INVESTIGATE
```
Recommended action:

```text
Investigate unusual pattern
```

Operations guidance:

    Review for less obvious unusual patterns.
    Compare payment amount, risk score, and channel against normal payments.
    Check whether the model is identifying a pattern not covered by business rules.
    Consider adding new business rules if repeated patterns are meaningful.

### RULE_ONLY

Meaning:

```text
Flagged by business rules but not by the Isolation Forest model
```
Recommended priority:

```text
P3_KNOWN_RISK
```
Recommended action:

```text
Review known business-risk pattern
```

Operations guidance:

    Review known risk indicators.
    Check the specific rule-based anomaly reasons.
    Determine whether the payment matches a known operational risk scenario.
    Continue processing only after the known risk is understood.

### NORMAL

Meaning

```text
No anomaly review required
```
Recommended priority:

```text
P4_NORMAL
```
Recommended action:

```text
No anomaly review needed
```

## Important Fields to Review

Operations should pay attention to:

    anomaly score
    anomaly band
    anomaly source
    review priority
    anomaly reasons
    payment amount
    risk-adjusted amount
    counterparty risk score
    payment type
    channel
    historical failure count

## Business Interpretation

An anomaly is not automatically fraud or failure.

It means the payment is unusual compared with expected patterns or matches a known business-risk rule.

The anomaly workflow should support prioritization and investigation.

## Current Limitations

This workflow is not production-ready.

Current limitations:

1. The model is trained on synthetic data.
2. Isolation Forest results are unsupervised.
3. Anomaly labels are not confirmed fraud or confirmed operational issues.
4. Business rules are simplified.
5. There is no human review feedback loop yet.
6. There is no real-time alerting.
7. There is no case management workflow.

## Production Recommendations

Before production use, the following improvements are required:

1. Validate anomaly patterns with operations experts.
2. Capture human review outcomes.
3. Add feedback loop for rule tuning.
4. Add case management integration.
5. Add alerting and escalation workflow.
6. Track anomaly false positives and confirmed issues.
7. Monitor anomaly volume over time.