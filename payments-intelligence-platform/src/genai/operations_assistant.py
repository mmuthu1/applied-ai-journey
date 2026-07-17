from src.genai.retrieve_context import retrieve_documents


def classify_question(question):
    question_lower = question.lower()

    if any(term in question_lower for term in ["failure", "fail", "failed", "high risk"]):
        return "payment_failure_review"

    if any(term in question_lower for term in ["anomaly", "unusual", "outlier", "flagged"]):
        return "anomaly_review"

    if any(term in question_lower for term in ["cash", "forecast", "liquidity"]):
        return "cash_forecast_review"

    if any(term in question_lower for term in ["drift", "monitor", "metadata", "version", "logging"]):
        return "model_monitoring_review"

    return "general_payment_operations"


def build_answer_from_question_type(question, question_type, retrieved_documents):
    top_document = retrieved_documents[0] if retrieved_documents else None

    source_text = "No supporting document found."
    if top_document:
        source_text = f"{top_document['title']} ({top_document['source_path']})"

    if question_type == "payment_failure_review":
        answer = f"""
Question:
{question}

Answer:
For a high-risk or elevated failure-risk payment, operations should treat the prediction as a triage signal, not as a final decision.

Recommended steps:
1. Review the predicted failure probability and risk band.
2. Check payment amount, payment type, channel, country, and currency.
3. Review counterparty risk score and historical failure count.
4. Check whether the payment is high value, international, SWIFT, or FILE-based.
5. If the payment is HIGH risk, review before release.
6. If the payment is MEDIUM risk, monitor or queue for secondary review.
7. Document the review decision for auditability.

Important limitation:
The current model is experimental and trained on synthetic data, so human review is still required.

Primary source:
{source_text}
"""
        return answer.strip()

    if question_type == "anomaly_review":
        answer = f"""
Question:
{question}

Answer:
For a payment anomaly, operations should review the anomaly source and priority before deciding the next action.

Recommended steps:
1. Check whether the payment was flagged by rules, the model, or both.
2. If the source is RULE_AND_MODEL, treat it as high priority.
3. If the source is MODEL_ONLY, investigate the unusual pattern.
4. If the source is RULE_ONLY, review the known business-risk reason.
5. Review anomaly score, anomaly band, amount, risk-adjusted amount, channel, and counterparty risk score.
6. Escalate if the payment appears suspicious, operationally risky, or unusual for the customer/channel.
7. Record the review outcome.

Important limitation:
An anomaly is not automatically fraud or failure. It means the payment is unusual or matches a known risk pattern.

Primary source:
{source_text}
"""
        return answer.strip()

    if question_type == "cash_forecast_review":
        answer = f"""
Question:
{question}

Answer:
For a cash forecast, operations should interpret the prediction as a planning signal for expected payment activity.

Recommended steps:
1. Review the predicted next-day total payment amount.
2. Check the forecast band: LOW, MEDIUM, or HIGH.
3. Compare the forecast against the rolling 7-day average.
4. For HIGH forecasts, review liquidity needs, staffing, and high-value payment queues.
5. For LOW forecasts, check for holidays, weekends, missing files, or unexpected payment delays.
6. For MEDIUM forecasts, continue normal cash operations planning.
7. Track forecast error over time when actual daily totals become available.

Important limitation:
The current model is experimental, trained on synthetic data, and does not include holidays or prediction intervals.

Primary source:
{source_text}
"""
        return answer.strip()

    if question_type == "model_monitoring_review":
        answer = f"""
Question:
{question}

Answer:
For model monitoring or drift questions, operations should review whether current payment patterns differ from the reference data used by the model.

Recommended steps:
1. Check the drift summary for LOW, MEDIUM, or HIGH drift.
2. Identify which feature drifted, such as amount, risk score, payment type, channel, country, or currency.
3. Determine whether the shift is expected based on business activity.
4. If drift is HIGH, investigate before relying heavily on model outputs.
5. Review prediction logs to confirm which model version produced recent predictions.
6. Check the model metadata registry for artifact path, version, serving status, and known limitations.
7. Escalate persistent drift to the model owner.

Important limitation:
The current monitoring layer uses file-based logs and simulated scoring data. It is not a production monitoring system yet.

Primary source:
{source_text}
"""
        return answer.strip()

    answer = f"""
Question:
{question}

Answer:
This appears to be a general payment operations question.

Recommended steps:
1. Review the most relevant runbook or report.
2. Identify whether the question relates to failure risk, anomaly review, cash forecasting, or model monitoring.
3. Use the model output as decision support, not as an automatic decision.
4. Document the operational review outcome.
5. Escalate unclear or high-risk cases.

Important limitation:
This assistant is experimental and uses local project documentation only.

Primary source:
{source_text}
"""
    return answer.strip()


def answer_question(question, top_k=3):
    retrieved_documents = retrieve_documents(question=question, top_k=top_k)
    question_type = classify_question(question)

    answer = build_answer_from_question_type(
        question=question,
        question_type=question_type,
        retrieved_documents=retrieved_documents,
    )

    return {
        "question": question,
        "question_type": question_type,
        "answer": answer,
        "retrieved_documents": retrieved_documents,
    }


def print_assistant_response(response):
    print("Operations Assistant Response")
    print("-----------------------------")
    print(response["answer"])

    print("\nRetrieved context")
    print("-----------------")

    for index, document in enumerate(response["retrieved_documents"], start=1):
        print(
            f"{index}. {document['title']} "
            f"({document['document_type']}) "
            f"- {document['source_path']} "
            f"- score={document['score']}"
        )


def run_sample_questions():
    questions = [
        "What should operations do for a high risk payment?",
        "How should we review a payment anomaly?",
        "What does high data drift mean?",
        "How should we interpret a cash forecast?",
    ]

    for question in questions:
        response = answer_question(question)
        print_assistant_response(response)
        print("\n" + "=" * 80 + "\n")


def main():
    run_sample_questions()


if __name__ == "__main__":
    main()
