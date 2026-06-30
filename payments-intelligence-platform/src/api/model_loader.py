from functools import lru_cache

from src.models.predict_failure import load_model


@lru_cache
def get_payment_failure_model():
    return load_model()