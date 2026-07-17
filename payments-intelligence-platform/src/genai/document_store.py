import json
import re
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_PATH = Path("data/genai/document_store.json")


DOCUMENT_PATHS = [
    Path("docs/runbooks/payment_failure_review.md"),
    Path("docs/runbooks/anomaly_review.md"),
    Path("docs/runbooks/cash_forecast_review.md"),
    Path("docs/runbooks/model_monitoring_review.md"),
    Path("reports/payment_failure_model_evaluation.md"),
    Path("reports/cash_forecasting_model_evaluation.md"),
    Path("reports/payment_anomaly_detection_report.md"),
    Path("reports/payment_failure_model_v2_evaluation.md"),
    Path("reports/model_monitoring_report.md"),
    Path("README.md"),
]


def read_markdown_file(path):
    if not path.exists():
        print(f"Skipping missing document: {path}")
        return None

    with path.open("r", encoding="utf-8") as file:
        return file.read()


def extract_title(content, fallback_title):
    for line in content.splitlines():
        cleaned_line = line.strip()

        if cleaned_line.startswith("# "):
            return cleaned_line.replace("# ", "", 1).strip()

    return fallback_title


def classify_document(path):
    path_text = str(path)

    if "docs/runbooks" in path_text:
        return "runbook"

    if "reports" in path_text:
        return "report"

    if path.name == "README.md":
        return "readme"

    return "other"


def normalize_text(content):
    content = content.replace("\r\n", "\n")
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def count_words(content):
    return len(re.findall(r"\b\w+\b", content))


def build_document_record(path):
    content = read_markdown_file(path)

    if content is None:
        return None

    normalized_content = normalize_text(content)
    title = extract_title(
        content=normalized_content,
        fallback_title=path.stem.replace("_", " ").title(),
    )

    document_id = path.stem.lower().replace(" ", "_")

    return {
        "document_id": document_id,
        "title": title,
        "source_path": str(path),
        "document_type": classify_document(path),
        "word_count": count_words(normalized_content),
        "content": normalized_content,
    }


def build_document_store():
    generated_at = datetime.now(timezone.utc).isoformat()
    documents = []

    for path in DOCUMENT_PATHS:
        document = build_document_record(path)

        if document is not None:
            documents.append(document)

    return {
        "generated_at": generated_at,
        "document_count": len(documents),
        "documents": documents,
    }


def save_document_store(document_store):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(document_store, file, indent=2)

    print("Saved local document store")
    print("--------------------------")
    print(f"Path: {OUTPUT_PATH}")
    print(f"Document count: {document_store['document_count']}")


def print_document_store_summary(document_store):
    print("\nDocument store summary")
    print("----------------------")

    for document in document_store["documents"]:
        print(
            f"{document['document_id']} | "
            f"type={document['document_type']} | "
            f"words={document['word_count']} | "
            f"path={document['source_path']}"
        )


def main():
    document_store = build_document_store()
    save_document_store(document_store)
    print_document_store_summary(document_store)


if __name__ == "__main__":
    main()
