# Bike Sharing Demand Prediction (ML Zoomcamp Midterm Project)

This project builds an automated machine learning service to predict hourly bike-sharing rental demand based on calendar information, such as hour, day of week, month, season, and holiday/working-day status.

---

## 📋 Problem Description
Urban bike-sharing systems face constant supply and demand imbalances. At the end of each day, predict the number of rentals for each hour of the following day allows fleet operators and city planners to proactively optimize bike availability, prevent station stockouts, and manage rebalancing logistics.

* **Target Variable:** `cnt` (Total count of rental bikes including both casual and registered users)
* **Model Type:** Regression (Supervised Learning)
* **Evaluation Metric:** Root Mean Squared Error (RMSE) and Root Mean Squared Log Error (RMSLE) are candidate metrics. The final evaluation metric will be selected and justified after establishing a baseline.

---

## 🗂️ Dataset
The project uses the **UCI Bike Sharing Dataset (`hour.csv`)**, which spans 2 years of historical data from the Capital Bikeshare system.
* **Features Include:**
  * **Temporal:** Hour of the day (`hr`), day of the week, working day, holiday, season, month.
  * **Environmental/Weather:** Normalized temperature (`temp`), apparent temperature (`atemp`), humidity (`hum`), wind speed (`windspeed`), and weather condition (`weathersit`).

The dataset contains observed weather, not next-day weather forecasts, so the first model will not use those observed weather values to predict tomorrow.

---

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Dependency Management:** Pipenv
* **Machine Learning:** Scikit-Learn, XGBoost, Pandas, NumPy
* **Model Serialization:** Pickle
* **Web Service:** FastAPI + Uvicorn
* **Containerization:** Docker

---

## 📁 Repository Structure
```text
├── data/                 # Raw dataset (hour.csv)
├── models/               # Saved model binaries (.bin)
├── Dockerfile            # Container definition
├── notebook.ipynb        # Jupyter notebook for EDA and model tuning
├── Pipfile               # Dependency definitions
├── Pipfile.lock          # Locked dependency versions
├── predict.py            # FastAPI prediction web service
├── train.py              # Script to train and persist the best model
└── README.md             # Project documentation
```

---

## 🚀 Usage

### 1. Install dependencies

```bash
pipenv install
```

### 2. Train the model

```bash
pipenv run python train.py
```

This script:
* loads `data/hour.csv`,
* builds a chronological train / validation / test split,
* compares baselines, linear regression, decision tree, random forest and tuned XGBoost,
* saves the final model to `models/bike_demand_model.bin`.

### 3. Run the prediction service locally

```bash
pipenv run uvicorn predict:app --host 0.0.0.0 --port 8000
```

Send a request with curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "hours": [
      {"season": 4, "yr": 1, "mnth": 10, "hr": 0, "holiday": 0, "weekday": 1, "workingday": 1},
      {"season": 4, "yr": 1, "mnth": 10, "hr": 8, "holiday": 0, "weekday": 1, "workingday": 1}
    ]
  }'
```

Expected response:

```json
{"predictions": [44, 725]}
```

### 4. Run with Docker

Build the image and start the container:

```bash
docker build -t bike-sharing-prediction .
docker run -p 8000:8000 bike-sharing-prediction
```

The service is then available at `http://localhost:8000`.