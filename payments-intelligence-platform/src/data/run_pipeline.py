from pathlib import Path

from src.data.clean_payments import clean_payments
from src.data.feature_engineering import create_features
from src.data.generate_payments import generate_payments, print_dataset_summary, NUM_RECORDS
from src.data.validate_payments import run_validation


RAW_OUTPUT_PATH = Path("data/raw/payments_raw.csv")
CLEAN_OUTPUT_PATH = Path("data/processed/payments_clean.csv")
FEATURES_OUTPUT_PATH = Path("data/processed/payments_features.csv")


def ensure_directories_exist():
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def generate_raw_data():
    print("\nSTEP 1: Generating raw payment data")
    print("-----------------------------------")

    df = generate_payments(NUM_RECORDS)
    df.to_csv(RAW_OUTPUT_PATH, index=False)

    print(f"Generated raw dataset: {RAW_OUTPUT_PATH}")
    print_dataset_summary(df)


def validate_raw_data():
    print("\nSTEP 2: Validating raw payment data")
    print("----------------------------------")

    run_validation(RAW_OUTPUT_PATH)


def clean_data():
    print("\nSTEP 3: Cleaning payment data")
    print("----------------------------")

    clean_payments(RAW_OUTPUT_PATH, CLEAN_OUTPUT_PATH)


def validate_clean_data():
    print("\nSTEP 4: Validating clean payment data")
    print("------------------------------------")

    run_validation(CLEAN_OUTPUT_PATH)


def engineer_features():
    print("\nSTEP 5: Engineering ML features")
    print("-------------------------------")

    create_features(CLEAN_OUTPUT_PATH, FEATURES_OUTPUT_PATH)


def validate_feature_data():
    print("\nSTEP 6: Validating final feature dataset")
    print("---------------------------------------")

    feature_path = FEATURES_OUTPUT_PATH

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {feature_path}")

    print(f"Feature dataset exists: {feature_path}")


def run_pipeline():
    ensure_directories_exist()
    generate_raw_data()
    validate_raw_data()
    clean_data()
    validate_clean_data()
    engineer_features()
    validate_feature_data()

    print("\nPipeline completed successfully")


def main():
    run_pipeline()


if __name__ == "__main__":
    main()