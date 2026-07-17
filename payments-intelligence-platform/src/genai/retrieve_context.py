import json
import re
from pathlib import Path


DOCUMENT_STORE_PATH = Path("data/genai/document_store.json")


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "mean",
    "of",
    "on",
    "or",
    "should",
    "the",
    "this",
    "to",
    "we",
    "what",
    "when",
    "with",
    "both",
    "made",
    "prediction",
}


def load_document_store():
    if not DOCUMENT_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Document store not found: {DOCUMENT_STORE_PATH}. "
            "Run python -m src.genai.document_store first."
        )

    with DOCUMENT_STORE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def tokenize(text):
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]+\b", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def score_document(query_tokens, document):
    content = document["content"].lower()
    title = document["title"].lower()
    source_path = document["source_path"].lower()

    score = 0
    matched_terms = []

    for token in query_tokens:
        title_matches = title.count(token)
        content_matches = content.count(token)
        path_matches = source_path.count(token)

        token_score = (
            title_matches * 5
            + path_matches * 3
            + content_matches
        )

        if token_score > 0:
            matched_terms.append(token)
            score += token_score
    
    query_text = " ".join(query_tokens)
    document_id = document["document_id"]

    if "rules" in query_text and "model" in query_text:
        if document_id == "anomaly_review":
            score += 60

    if "flagged" in query_text:
        if document_id == "anomaly_review":
            score += 40

    if "forecasted" in query_text or "forecast" in query_text:
        if document_id == "cash_forecast_review":
            score += 60

    if "volume" in query_text:
        if document_id == "cash_forecast_review":
            score += 40

    # Prefer runbooks for operational questions.
    if document["document_type"] == "runbook":
        score = score * 2.5

    # Reports are useful supporting context, but should not outrank runbooks
    # for operational guidance questions.
    elif document["document_type"] == "report":
        score = score * 0.75

    # README is broad and should not dominate operational retrieval.
    elif document["document_type"] == "readme":
        score = score * 0.4

    return score, sorted(set(matched_terms))


def retrieve_documents(question, top_k=3):
    document_store = load_document_store()
    query_tokens = tokenize(question)

    scored_documents = []

    for document in document_store["documents"]:
        score, matched_terms = score_document(
            query_tokens=query_tokens,
            document=document,
        )

        if score > 0:
            scored_documents.append(
                {
                    "document_id": document["document_id"],
                    "title": document["title"],
                    "source_path": document["source_path"],
                    "document_type": document["document_type"],
                    "score": round(score, 2),
                    "matched_terms": matched_terms,
                    "content_preview": document["content"][:700],
                }
            )

    scored_documents = sorted(
        scored_documents,
        key=lambda item: item["score"],
        reverse=True,
    )

    return scored_documents[:top_k]


def print_retrieval_results(question, results):
    print("Retrieval question")
    print("------------------")
    print(question)

    print("\nRetrieved documents")
    print("-------------------")

    if not results:
        print("No relevant documents found.")
        return

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("--------")
        print(f"Document: {result['title']}")
        print(f"Type: {result['document_type']}")
        print(f"Path: {result['source_path']}")
        print(f"Score: {result['score']}")
        print(f"Matched terms: {', '.join(result['matched_terms'])}")

        print("\nPreview:")
        print(result["content_preview"])


def run_sample_questions():
    questions = [
        "What should operations do for a high risk payment?",
        "How should we review a payment anomaly?",
        "What does high data drift mean?",
        "How should we interpret a cash forecast?",
    ]

    for question in questions:
        results = retrieve_documents(question=question, top_k=3)
        print_retrieval_results(question=question, results=results)
        print("\n" + "=" * 80 + "\n")


def main():
    run_sample_questions()


if __name__ == "__main__":
    main()
