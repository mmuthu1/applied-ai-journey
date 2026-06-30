from functools import lru_cache

from src.forecasting.predict_cash_forecast import load_model_artifact
from src.models.predict_failure import load_model


@lru_cache
def get_payment_failure_model():
    return load_model()


@lru_cache
def get_cash_forecast_artifact():
    return load_model_artifact()