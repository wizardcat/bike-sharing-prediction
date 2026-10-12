# Bike Sharing Demand Prediction (ML Zoomcamp Midterm Project)

This project builds an automated machine learning service to predict hourly bike-sharing rental demand based on environmental conditions, seasonal factors, and temporal variables.

---

## 📋 Problem Description
Urban bike-sharing systems face constant supply and demand imbalances. Predicting the number of rented bikes (`cnt`) for any given hour allows fleet operators and city planners to proactively optimize bike availability, prevent station stockouts, and manage rebalancing logistics.

* **Target Variable:** `cnt` (Total count of rental bikes including both casual and registered users)
* **Model Type:** Regression (Supervised Learning)
* **Evaluation Metric:** Root Mean Squared Error (RMSE) / Root Mean Squared Log Error (RMSLE)

---

## 🗂️ Dataset
The project uses the **UCI Bike Sharing Dataset (`hour.csv`)**, which spans 2 years of historical data from the Capital Bikeshare system.
* **Features Include:**
  * **Temporal:** Hour of the day (`hr`), day of the week, working day, holiday, season, month.
  * **Environmental/Weather:** Normalized temperature (`temp`), apparent temperature (`atemp`), humidity (`hum`), wind speed (`windspeed`), and weather condition (`weathersit`).

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
├── notebooks/            # Jupyter notebooks for EDA and model tuning
├── models/               # Saved model binaries (.bin)
├── Dockerfile            # Container definition
├── Pipfile               # Dependency definitions
├── Pipfile.lock          # Locked dependency versions
├── predict.py            # FastAPI prediction web service
├── train.py              # Script to train and persist the best model
└── README.md             # Project documentation