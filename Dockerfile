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

# Copy the prediction code (features.py is shared with train.py) and the
# trained model artifact.
COPY features.py predict.py ./
COPY models/bike_demand_model.bin ./models/

# Render (and most PaaS platforms) inject a PORT to bind to; 8000 is the
# default for local `docker run -p 8000:8000`.
ENV PORT=8000
EXPOSE 8000

# exec so uvicorn receives SIGTERM directly and shuts down cleanly.
ENTRYPOINT ["sh", "-c", "exec uvicorn predict:app --host 0.0.0.0 --port ${PORT}"]
