FROM python:3.11-slim

WORKDIR /app

# Copy model artifacts and serving code
COPY models/     models/
COPY serving/    serving/
COPY data/       data/
COPY requirements.txt .

# Install only runtime dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pandas \
    numpy \
    scikit-learn \
    xgboost \
    joblib \
    pyarrow

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
