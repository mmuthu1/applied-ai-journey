from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.genai.operations_assistant import answer_question


OUTPUT_PATH = Path("reports/genai_assistant_evaluation.md")


EVALUATION_QUESTIONS = [
    {
        "question": "What should operations do for a high risk payment?",
        "expected_question_type": "payment_failure_review",
        "expected_top_document_id": "payment_failure_review",
    },
    {
        "question": "How should we review a payment anomaly?",
        "expected_question_type": "anomaly_review",
        "expected_top_document_id": "anomaly_review",
    },
    {
        "question": "What does high data drift mean?",
        "expected_question_type": "model_monitoring_review",
        "expected_top_document_id": "model_monitoring_review",
    },
    {
        "question": "How should we interpret a cash forecast?",
        "expected_question_type": "cash_forecast_review",
        "expected_top_document_id": "cash_forecast_review",
    },
    {
        "question": "Which model version made this prediction?",
        "expected_question_type": "model_monitoring_review",
        "expected_top_document_id": "model_monitoring_review",
    },
    {
        "question": "What should we do if a payment is flagged by both rules and model?",
        "expected_question_type": "anomaly_review",
        "expected_top_document_id": "anomaly_review",
    },
    {
        "question": "What should we check before releasing a high failure risk SWIFT payment?",
        "expected_question_type": "payment_failure_review",
        "expected_top_document_id": "payment_failure_review",
    },
    {
        "question": "What should we do when forecasted payment volume is high?",
        "expected_question_type": "cash_forecast_review",
        "expected_top_document_id": "cash_forecast_review",
    },
]


def evaluate_question(evaluation_case):
    response = answer_question(evaluation_case["question"])

    retrieved_documents = response["retrieved_documents"]
    top_document = retrieved_documents[0] if retrieved_documents else None

    actual_top_document_id = (
        top_document["document_id"]
        if top_document
        else None
    )

    actual_top_document_title = (
        top_document["title"]
        if top_document
        else None
    )

    question_type_passed = (
        response["question_type"]
        == evaluation_case["expected_question_type"]
    )

    top_document_passed = (
        actual_top_document_id
        == evaluation_case["expected_top_document_id"]
    )

    return {
        "question": evaluation_case["question"],
        "expected_question_type": evaluation_case["expected_question_type"],
        "actual_question_type": response["question_type"],
        "question_type_passed": question_type_passed,
        "expected_top_document_id": evaluation_case["expected_top_document_id"],
        "actual_top_document_id": actual_top_document_id,
        "actual_top_document_title": actual_top_document_title,
        "top_document_passed": top_document_passed,
        "overall_passed": question_type_passed and top_document_passed,
    }


def run_evaluation():
    results = []

    for evaluation_case in EVALUATION_QUESTIONS:
        result = evaluate_question(evaluation_case)
        results.append(result)

    return pd.DataFrame(results)


def build_report(results_df):
    generated_at = datetime.now(timezone.utc).isoformat()

    total_questions = len(results_df)
    passed_questions = int(results_df["overall_passed"].sum())
    failed_questions = total_questions - passed_questions

    pass_rate = passed_questions / total_questions if total_questions else 0

    question_type_pass_rate = (
        results_df["question_type_passed"].sum() / total_questions
        if total_questions
        else 0
    )

    top_document_pass_rate = (
        results_df["top_document_passed"].sum() / total_questions
        if total_questions
        else 0
    )

    report = f"""# GenAI Operations Assistant Evaluation

    Generated at:

    ```text
    {generated_at}
    ```

    Executive Summary

    This report evaluates the local GenAI/RAG-style operations assistant for the Payments Intelligence Platform.

    The evaluation checks whether the assistant:

    Classifies the question into the expected operations category.
    Retrieves the expected runbook as the top supporting document.
    Evaluation Summary
    | Metric                           |                         Value |
    | -------------------------------- | ----------------------------: |
    | Total questions                  |             {total_questions} |
    | Passed questions                 |            {passed_questions} |
    | Failed questions                 |            {failed_questions} |
    | Overall pass rate                |               {pass_rate:.2%} |
    | Question type pass rate          | {question_type_pass_rate:.2%} |
    | Top document retrieval pass rate |  {top_document_pass_rate:.2%} |

    Evaluation Results

    {results_df.to_markdown(index=False)}

    Interpretation

    This evaluation confirms whether the assistant can route common payment operations questions to the correct runbook.

    The current assistant uses local project documentation and simple retrieval logic. It does not call an external LLM.

    Current Limitations
    1. The assistant uses keyword-based retrieval, not embeddings.
    2. The assistant uses rule-based answer templates, not generative reasoning.
    3. The evaluation set is small.
    4. The document store is local JSON.
    5. The assistant has not been tested with real users.
    6. The assistant does not yet support follow-up questions.
    7. The assistant does not access live payment records or production systems.

    Recommended Next Improvements
    1. Add more evaluation questions.
    2. Add chunk-level retrieval instead of full-document retrieval.
    3. Add semantic embeddings.
    4. Add an LLM-based answer generation layer.
    5. Add source citations in final answers.
    6. Add feedback capture from operations users.
    7. Add guardrails for uncertain answers.
    """
    
    return report

def save_report(report):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        file.write(report)

    print("Saved GenAI assistant evaluation report")
    print("--------------------------------------")
    print(f"Path: {OUTPUT_PATH}")

def print_evaluation_summary(results_df):
    total_questions = len(results_df)
    passed_questions = int(results_df["overall_passed"].sum())
    failed_questions = total_questions - passed_questions
    print("GenAI assistant evaluation summary")
    print("----------------------------------")
    print(f"Total questions: {total_questions}")
    print(f"Passed questions: {passed_questions}")
    print(f"Failed questions: {failed_questions}")
    print(f"Overall pass rate: {passed_questions / total_questions:.2%}")

    print("\nDetailed results")
    print("----------------")
    print(
        results_df[
            [
                "question",
                "actual_question_type",
                "actual_top_document_id",
                "overall_passed",
            ]
        ].to_string(index=False)
    )

def main():
    results_df = run_evaluation()
    report = build_report(results_df)
    save_report(report)
    print_evaluation_summary(results_df)

if __name__ == "__main__":
    main()