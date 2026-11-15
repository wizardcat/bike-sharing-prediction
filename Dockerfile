# Dockerfile for the Bike Sharing Demand Prediction service.
# Builds a small container that runs the FastAPI app from predict.py.

FROM python:3.12.15-slim-bookworm

# XGBoost needs a C++ compiler and OpenMP runtime during installation.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies from the pinned requirements file.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the prediction script and the trained model artifact.
COPY predict.py .
COPY models/bike_demand_model.bin ./models/

# Expose the port that uvicorn will listen on.
EXPOSE 8000

# Start the FastAPI application.
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
