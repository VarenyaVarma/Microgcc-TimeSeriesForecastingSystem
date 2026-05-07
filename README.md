---
# 🚀 Time Series Forecasting System

**Candidate:** P. B. Varenya Varma | **UGID:** 22WU0101018 | **Role:** Data Science | **Company:** Microgcc

Production-ready end-to-end forecasting backend built to predict the next **8 weeks of sales** for multiple states using historical time-series data.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-green?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/XGBoost-ML-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/TensorFlow-LSTM-red?style=for-the-badge&logo=tensorflow" />
</p>

---

## 📌 Project Overview

This project was designed as a **real-world forecasting backend system** rather than a notebook-only implementation.

The system:

* preprocesses and validates historical sales data
* performs advanced feature engineering
* trains and compares multiple forecasting models
* automatically selects the best-performing model
* generates 8-week sales forecasts
* exposes predictions through a FastAPI backend
* provides Swagger API documentation for testing

---

## 🏗️ System Architecture

```text
Data Ingestion
      ↓
Preprocessing & Cleaning
      ↓
Feature Engineering
      ↓
Train / Validation Split
      ↓
Model Training
(ARIMA | Prophet | XGBoost | LSTM)
      ↓
Model Evaluation
(RMSE | MAE | MAPE)
      ↓
Best Model Selection
      ↓
Forecast Generation
      ↓
FastAPI REST Backend
```

---

## 🤖 Forecasting Models

| Model            | Purpose                              |
| ---------------- | ------------------------------------ |
| SARIMA           | Captures trend & seasonality         |
| Facebook Prophet | Business-oriented forecasting        |
| XGBoost          | Feature-driven ML forecasting        |
| LSTM             | Sequential deep learning forecasting |

---

## ⚙️ Feature Engineering

Implemented forecasting features include:

### Lag Features

* t-1
* t-7
* t-30

### Rolling Statistics

* Rolling Mean
* Rolling Standard Deviation

### Temporal Features

* Month
* Day of Week
* Weekend Flag
* Quarter

### Holiday Features

* Holiday Indicators
* Seasonal Awareness

### Validation Strategy

* Chronological train-validation split
* No data leakage

---

## 🧰 Tech Stack

```text
Python
Pandas
NumPy
Scikit-learn
Statsmodels
Prophet
XGBoost
TensorFlow / Keras
FastAPI
Matplotlib
```

---

## 📂 Project Structure

```bash
forecasting-system/
│
├── api/                 # FastAPI backend
├── data/                # Dataset
├── models/              # Saved trained models
├── outputs/             # Forecasts, plots, metrics
├── src/                 # Core forecasting pipeline
├── notebooks/           # EDA notebook
├── requirements.txt
├── run_pipeline.py
├── README.md
└── case_study_report.md
```

---

## ⚡ Installation & Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd forecasting-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Run Forecasting Pipeline

```bash
python run_pipeline.py
```

### Start FastAPI Backend

```bash
uvicorn api.app:app --reload
```

---

## 🌐 API Access

### Swagger Documentation

```txt
http://localhost:8000/docs
```

### Health Endpoint

```txt
http://localhost:8000/health
```

### Forecast Endpoint

```txt
/forecast?state=California
```

### Model Comparison Endpoint

```txt
/model-comparison?state=California
```

---

## 📊 Evaluation Metrics

| Metric | Full Form                      |
| ------ | ------------------------------ |
| RMSE   | Root Mean Squared Error        |
| MAE    | Mean Absolute Error            |
| MAPE   | Mean Absolute Percentage Error |

The system automatically selects the best-performing model based on validation RMSE.

---

## ✨ Key Highlights

* Production-style modular architecture
* Automated model comparison
* Time-series aware validation
* FastAPI backend integration
* Swagger-based API testing
* Forecast visualization support
* Multiple forecasting paradigms
* Real-world backend structure

---

## 🔮 Future Improvements

* Docker deployment
* CI/CD integration
* Ensemble forecasting
* Automated retraining pipeline
* Cloud deployment support
* Real-time monitoring

---

## 👤 Author

**P. B. Varenya Varma**
B.Tech Computer Science Engineering
Data Science | Full Stack Development | AI/ML
