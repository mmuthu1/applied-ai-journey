# Payment Failure Review Runbook

## Purpose

This runbook explains how to review payments that are predicted to have elevated failure risk.

The payment failure classifier estimates whether a payment may fail based on pre-outcome payment attributes such as amount, payment type, channel, country, counterparty risk score, settlement window, and historical failure patterns.

## Model Used

Current improved model:

```text
payment_failure_classifier_v2
```

Current recommended threshold:

```text
0.55
```
The v2 model is an improved experimental classifier. It reduced false positives compared with the original baseline while keeping useful recall.

## Risk Bands
The inference workflow assigns a prediction risk band.

### LOW Risk
Typical action:

```text
Allow normal processing
```

Operations guidance:

    Continue normal processing.
    No immediate manual review is required.
    Monitor only if other business rules trigger concern.

### MEDIUM Risk
Typical action:

```text
Monitor or queue for secondary review
```

Operations guidance:

    Review if payment amount is high.
    Review if the payment uses SWIFT or FILE channel.
    Review if the counterparty risk score is elevated.
    Review if the counterparty has prior failures.
    Consider secondary review before release.

### HIGH Risk
Typical action:

```text
Review before release
```

Operations guidance:

    Prioritize for manual review.
    Check counterparty risk score.
    Check historical failure count.
    Check payment type and channel.
    Check settlement window.
    Confirm whether the payment is high value or international.
    Consider holding or escalating before release.

## Important Features to Review

Operations should pay close attention to:

    payment amount
    payment type
    payment channel
    country
    currency
    counterparty risk score
    historical failure count
    settlement window
    risk-adjusted amount

## Business Interpretation

A high predicted failure probability does not mean the payment will definitely fail.

It means the payment has characteristics similar to payments that failed in the synthetic training data.

The model should support operational triage, not replace human decision-making.

## Current Limitations

This model is not production-ready.

Current limitations:

1. The model is trained on synthetic data.
2. The v2 features include dataset-level historical approximations.
3. Production usage would require time-aware historical features.
4. The model is not calibrated.
5. The model has not been validated on real payment data.
6. The current prediction log is file-based.
7. There is no human review outcome feedback loop yet.

## Production Recommendations
Before production use, the following improvements are required:

1. Validate on real historical payment data.
2. Add time-aware historical counterparty features.
3. Calibrate predicted probabilities.
4. Add human review outcome capture.
5. Monitor precision, recall, false positives, and false negatives over time.
6. Add model approval and rollback workflow.
7. Store predictions and review outcomes in a database.