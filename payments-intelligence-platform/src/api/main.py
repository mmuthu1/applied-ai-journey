from fastapi import FastAPI


app = FastAPI(
    title="Payments Intelligence Platform API",
    description=(
        "API for payment failure prediction, cash forecasting, "
        "and payment anomaly detection."
    ),
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "Payments Intelligence Platform API",
        "status": "running",
        "available_endpoints": [
            "/health",
            "/docs",
        ],
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "payments-intelligence-platform-api",
        "version": "0.1.0",
    }
