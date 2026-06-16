# Payment Anomaly Detection Report

## 1. Executive Summary

This report evaluates the first payment anomaly detection workflow built for the Payments Intelligence Platform.

The objective is to identify unusual payment transactions using both business-rule anomaly logic and an unsupervised machine learning model.

The current anomaly detection approach combines:

```text
Business-rule anomaly detection
+
Isolation Forest anomaly detection
```

This combined approach is useful because business rules capture known risk patterns, while the model can detect unusual combinations that were not explicitly encoded in rules.

Key findings:

* The anomaly dataset contains 5,000 payment records.
* Rule-based logic flagged 183 payments as anomalies.
* Isolation Forest flagged 200 payments as anomalies.
* 141 payments were flagged by both rules and the model.
* Combined, 242 payments require anomaly review.
* The total anomaly review workload is 4.84%.
* Anomaly groups showed higher failure rates than normal payments.
* The workflow is useful as an experimental anomaly triage design, but it is not yet production-ready.

## 2. Business Problem

Payment operations teams need to detect unusual payment behavior before it becomes an operational issue.

Anomalous payments may include:

* Extremely large payments
* High risk-adjusted amount payments
* Large payments submitted through file channels
* High-risk SWIFT payments
* Payments involving prior failure history
* Unusual combinations of amount, risk, payment type, and channel

The goal is not only to identify anomalies, but also to route them into meaningful review queues.

A practical anomaly workflow should answer:

```text
Which payments should be reviewed first, and why?
```

## 3. Input Dataset

The anomaly workflow uses:

```text
data/processed/payment_anomaly_features.csv
```

This dataset was created from the ML-ready payment feature dataset:

```text
data/processed/payments_features.csv
```

Dataset summary:

```text
Rows: 5,000
Anomaly feature columns: 45
```

The scored anomaly dataset is saved as:

```text
data/processed/payment_anomaly_scored.csv
```

## 4. Rule-Based Anomaly Logic

The rule-based anomaly layer uses manually defined business-risk patterns.

Rule features include:

| Rule Feature                      | Meaning                                      |
| --------------------------------- | -------------------------------------------- |
| `is_extreme_amount`               | Payment amount has extreme z-score           |
| `is_top_1pct_amount`              | Payment amount is in the top 1%              |
| `is_extreme_risk_adjusted_amount` | Risk-adjusted amount has extreme z-score     |
| `is_top_1pct_risk_score`          | Counterparty risk score is in the top 1%     |
| `is_high_risk_large_payment`      | Large payment with high risk score           |
| `is_large_file_payment`           | Large payment submitted through FILE channel |
| `is_high_risk_swift_payment`      | SWIFT payment with high risk score           |
| `is_prior_failure_high_value`     | High-value payment with prior failures       |

Each payment receives:

```text
rule_anomaly_score
```

and a flag:

```text
is_rule_based_anomaly
```

Rule-based anomaly result:

```text
Rule-based anomalies: 183
Rule-based anomaly rate: 3.66%
```

## 5. Isolation Forest Model

The unsupervised anomaly detection model uses:

```text
Isolation Forest
```

Isolation Forest learns patterns in the payment data and flags records that are easier to isolate from the rest of the population.

The model was configured with:

```text
contamination = 0.04
```

This asks the model to flag roughly 4% of records as anomalies, which aligns with the rule-based anomaly rate of approximately 3.66%.

Isolation Forest result:

```text
Isolation Forest anomalies: 200
```

## 6. Rule-Based vs Model-Based Comparison

The rule-based and model-based anomaly results were compared.

| Category       | Count | Meaning                                             |
| -------------- | ----: | --------------------------------------------------- |
| Normal         | 4,758 | Not flagged by rule or model                        |
| Rule and model |   141 | Flagged by both business rules and Isolation Forest |
| Model only     |    59 | Flagged only by Isolation Forest                    |
| Rule only      |    42 | Flagged only by business rules                      |

Summary:

```text
Rule-based anomalies: 183
Isolation Forest anomalies: 200
Overlap anomalies: 141
Combined anomalies: 242
Combined anomaly rate: 4.84%
Rule anomaly coverage by model: 77.05%
Model anomaly overlap with rules: 70.50%
```

This is a strong result because the model agrees with many business-rule anomalies while also discovering additional unusual patterns.

## 7. Review Queue Design

The workflow assigns each payment to an anomaly source:

| Anomaly Source   | Count | Review Priority  | Recommended Action          |
| ---------------- | ----: | ---------------- | --------------------------- |
| `RULE_AND_MODEL` |   141 | `P1_HIGH`        | High-priority review        |
| `MODEL_ONLY`     |    59 | `P2_INVESTIGATE` | Investigate unusual pattern |
| `RULE_ONLY`      |    42 | `P3_KNOWN_RISK`  | Review known risk pattern   |
| `NORMAL`         | 4,758 | `P4_NORMAL`      | No anomaly review needed    |

Review workload:

```text
Payments requiring anomaly review: 242
Review workload rate: 4.84%
```

This is a practical operational queue size for an experimental anomaly detection workflow.

## 8. Failure Rate by Anomaly Source

The anomaly groups showed higher failure rates than normal payments.

| Anomaly Source   | Payment Count | Failed Count | Failure Rate |
| ---------------- | ------------: | -----------: | -----------: |
| `RULE_ONLY`      |            42 |            5 |       11.90% |
| `MODEL_ONLY`     |            59 |            7 |       11.86% |
| `RULE_AND_MODEL` |           141 |           12 |        8.51% |
| `NORMAL`         |         4,758 |          257 |        5.40% |

This is important because anomaly groups are not just unusual — they also show elevated failure risk compared with normal payments.

## 9. Top Anomaly Reasons

The most common anomaly reasons include:

* `is_top_1pct_risk_score`
* `is_extreme_amount`
* `is_top_1pct_amount`
* `is_extreme_risk_adjusted_amount`
* `is_large_file_payment`
* `is_prior_failure_high_value`
* `is_high_risk_swift_payment`

These reasons align with payment operations intuition because unusually large payments, high-risk counterparties, file-based submissions, and prior failure history can all increase operational risk.

## 10. Operational Interpretation

The anomaly workflow should be interpreted as a triage system, not as an automatic decision system.

Recommended triage design:

| Queue            | Meaning               | Suggested Use                        |
| ---------------- | --------------------- | ------------------------------------ |
| `P1_HIGH`        | Rules and model agree | First review queue                   |
| `P2_INVESTIGATE` | Model-only anomaly    | Discovery queue for unusual patterns |
| `P3_KNOWN_RISK`  | Rule-only anomaly     | Known-risk business rule queue       |
| `P4_NORMAL`      | No anomaly            | No anomaly review needed             |

The most valuable part of this design is that each anomaly has a reason and a recommended action.

## 11. Why Combine Rules and ML?

Rules and unsupervised ML are complementary.

Business rules are useful because they capture known risk patterns, such as:

* Top percentile risk scores
* Extreme payment amounts
* Large FILE payments
* High-risk SWIFT payments

Isolation Forest is useful because it can detect unusual combinations that were not explicitly written as rules.

A combined workflow is stronger than either approach alone:

```text
Rules provide business control.
ML provides pattern discovery.
```

## 12. Current Status

Current status:

```text
Experimental anomaly detection workflow
```

Not yet:

```text
Production-ready payment anomaly monitoring system
```

The workflow is useful for learning, portfolio demonstration, and early anomaly triage design.

## 13. Limitations

Current limitations:

1. The dataset is synthetic.
2. The anomaly rules are manually designed.
3. Isolation Forest was not tuned extensively.
4. The contamination rate is fixed at 4%.
5. There is no human feedback loop yet.
6. The model does not include real customer/counterparty behavior history.
7. There is no alert suppression logic.
8. There is no case-management workflow.
9. There is no model monitoring or drift detection.
10. Anomalies are not validated against real investigation outcomes.

## 14. Recommended Next Improvements

Recommended improvements:

1. Tune Isolation Forest contamination rate.
2. Add counterparty-level historical behavior features.
3. Add rolling amount and volume features by counterparty.
4. Add currency-pair and country-pair anomaly features.
5. Add alert suppression rules to reduce repeated alerts.
6. Add human feedback labels from analyst review.
7. Track anomaly outcomes and investigation results.
8. Create model artifact saving script.
9. Create anomaly inference script for new payments.
10. Add API endpoint for anomaly scoring.
11. Add monitoring and drift detection.

## 15. Conclusion

The anomaly detection workflow successfully demonstrates a practical applied AI pattern:

```text
business-rule anomaly logic
→ unsupervised anomaly detection
→ rule/model comparison
→ review queue prioritization
→ business interpretation
```

The most important finding is that anomaly groups show higher failure rates than normal payments.

This supports the idea that a combined rule and model anomaly workflow can help payment operations teams prioritize unusual and potentially risky payments.

The key applied AI lesson is:

```text
Anomaly detection should not only flag unusual records; it should explain why they matter and how operations should respond.
```