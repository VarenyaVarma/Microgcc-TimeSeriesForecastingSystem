# Time Series Forecasting System

**Production-Grade Multi-Model Sales Forecasting Pipeline**

**Candidate:** P. B. Varenya Varma | **UGID:** 22WU0101018 | **Role:** Data Science | **Company:** Microgcc

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Folder Structure](#folder-structure)
4. [Features](#features)
5. [Models Implemented](#models-implemented)
6. [Feature Engineering](#feature-engineering)
7. [Setup Instructions](#setup-instructions)
8. [Installation](#installation)
9. [Running the Pipeline](#running-the-pipeline)
10. [Running the API](#running-the-api)
11. [API Documentation](#api-documentation)
12. [Example API Calls](#example-api-calls)
13. [Evaluation Metrics](#evaluation-metrics)
14. [Key Assumptions](#key-assumptions)
15. [Future Enhancements](#future-enhancements)

---

## Project Overview

This is a **production-ready end-to-end time series forecasting system** designed to:

- Train and compare multiple state-of-the-art forecasting algorithms
- Automatically select the best model based on validation performance
- Generate accurate 8-week sales forecasts for multiple geographic states
- Expose predictions via a scalable REST API with comprehensive monitoring
- Handle real-world data challenges: missing dates, missing values, seasonality, and trends
- Follow enterprise-grade software engineering standards

### Key Objectives

✅ **Multi-Model Comparison:** ARIMA, Prophet, XGBoost, LSTM  
✅ **Automated Model Selection:** Best model chosen based on RMSE  
✅ **Feature-Rich Engineering:** Lag features, rolling features, date features, holidays  
✅ **Temporal Validation:** No data leakage, proper time-series train-test splits  
✅ **Scalable Architecture:** Modular design for production deployment  
✅ **REST API:** FastAPI with Swagger documentation  
✅ **Professional Documentation:** README, case study report, inline code comments  

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│           Data Ingestion & Preprocessing                │
│  (handle missing dates, missing values, outliers)       │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│         Feature Engineering Pipeline                    │
│  (lags, rolling, date features, holidays)               │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│      Temporal Train-Test Split (NO data leakage)        │
│  Train: 01-Jan-2022 → 31-Dec-2023                       │
│  Test: 01-Jan-2024 → 29-Feb-2024                        │
└────────────────────┬────────────────────────────────────┘
                     ↓
   ┌────────────────┬────────────────┬─────────────┐
   ↓                ↓                ↓             ↓
 ARIMA          Prophet         XGBoost         LSTM
 Model          Model           Model          Model
   │                │                │             │
   └────────────────┴────────────────┴─────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│      Model Evaluation & Selection                        │
│  (MAE, RMSE, MAPE comparison per state)                 │
│  Best model selected by RMSE                             │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│      Forecasting (8 weeks ahead)                        │
│  Using best model or ensemble                            │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│      REST API & Visualization                            │
│  FastAPI with Swagger docs                              │
│  Forecast plots, metrics charts                         │
└─────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
forecasting-system/
│
├── data/                              # Raw and processed data
│   └── sales_data.py                  # Dataset generation script
│
├── notebooks/                         # Jupyter notebooks
│   └── exploration.ipynb              # EDA and analysis
│
├── src/                               # Source code modules
│   ├── __init__.py
│   ├── config.py                      # Configuration settings
│   ├── utils.py                       # Utility functions (logging, metrics)
│   ├── preprocessing.py               # Data cleaning and preparation
│   ├── feature_engineering.py         # Feature creation
│   ├── train_models.py               # Model training (ARIMA, Prophet, XGBoost, LSTM)
│   ├── evaluate.py                   # Model evaluation metrics
│   └── predict.py                    # Prediction generation
│
├── models/                            # Trained model artifacts
│   ├── state_arima_model.pkl
│   ├── state_prophet_model.pkl
│   ├── state_xgboost_model.pkl
│   └── state_lstm_model.pkl
│
├── api/                               # REST API
│   ├── __init__.py
│   └── app.py                        # FastAPI application
│
├── outputs/                           # Pipeline outputs
│   ├── forecasts/                     # CSV forecasts per state
│   ├── plots/                         # Visualization plots
│   └── metrics/                       # JSON evaluation metrics
│
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── run_pipeline.py                    # Main execution script
├── case_study_report.md              # Detailed case study report
└── .gitignore                         # Git ignore rules
```

---

## Features

### 1. **Data Preprocessing Pipeline**
- **Missing Date Handling:** Complete date range generation and interpolation
- **Missing Value Imputation:** Forward fill, interpolation, and statistical methods
- **Outlier Detection:** Percentile-based detection and capping
- **Data Sorting:** Chronological ordering by state and date
- **Quality Assurance:** Comprehensive data quality reporting

### 2. **Feature Engineering**
- **Lag Features:** Sales at t-1, t-7, t-30
- **Rolling Statistics:** 7-day and 30-day rolling mean and std dev
- **Date Features:** Day of week, day of month, month, quarter, day of year, week of year
- **Cyclical Features:** Sine/cosine encoding for seasonal patterns
- **Holiday Features:** Holiday flags, days to/since holidays
- **State Grouping:** Separate feature sets per state

### 3. **Multiple Model Implementations**
- **ARIMA/SARIMA:** Autoregressive integrated moving average
- **Prophet:** Facebook's time series forecasting algorithm
- **XGBoost:** Gradient boosting with lag features
- **LSTM:** Deep learning sequential model with Keras

### 4. **Robust Evaluation**
- **Metrics:** MAE, RMSE, MAPE per model
- **Comparison:** Systematic comparison across all models
- **Automatic Selection:** Best model chosen by lowest RMSE
- **Ensembling:** Fallback ensemble averaging if needed

### 5. **Production-Ready API**
- **REST Endpoints:** Health check, state listing, forecasting, comparison
- **Request Validation:** Pydantic models for type safety
- **Error Handling:** Comprehensive error responses
- **Documentation:** Auto-generated Swagger/OpenAPI docs
- **Batch Operations:** Forecast all states in one request

### 6. **Visualization**
- **Forecast Plots:** Historical data + forecast with confidence intervals
- **Model Comparison:** RMSE and MAE across states and models
- **Professional Output:** High-resolution PNG plots

---

## Models Implemented

### 1. ARIMA/SARIMA
**Purpose:** Capture temporal dependencies and seasonality  
**Strengths:** Interpretable, handles trends well  
**Configuration:** Auto-detected or manual tuning per state  
**Training:** Up to 1000 iterations with stationarity enforcement

### 2. Facebook Prophet
**Purpose:** Robust to missing data and outliers  
**Strengths:** Built-in holiday handling, fast inference  
**Configuration:** Yearly and weekly seasonality enabled  
**Confidence Intervals:** 95% by default

### 3. XGBoost
**Purpose:** Capture non-linear patterns  
**Strengths:** Fast training, handles lag features effectively  
**Configuration:**
- 200 estimators, max depth 7
- Learning rate: 0.1
- Subsample: 0.8
- Early stopping: 50 rounds
**Input:** Engineered features including lags

### 4. LSTM (Deep Learning)
**Purpose:** Learn complex sequential patterns  
**Strengths:** Handles long dependencies  
**Architecture:**
- Input: 30-day sequence
- Layer 1: LSTM (128 units) + Dropout (0.2)
- Layer 2: LSTM (64 units) + Dropout (0.2)
- Dense layer: 32 units
- Output: 1 value
**Training:** 100 epochs, batch size 32, early stopping

---

## Feature Engineering

### Lag Features
```
sales_lag_1：t-1 (yesterday)
sales_lag_7：t-7 (one week ago)
sales_lag_30：t-30 (one month ago)
```

### Rolling Features
```
sales_rolling_mean_7：7-day average
sales_rolling_std_7：7-day volatility
sales_rolling_mean_30：30-day average
sales_rolling_std_30：30-day volatility
```

### Date Features
```
day_of_week：0-6 (Monday-Sunday)
day_of_month：1-31
month：1-12
quarter：1-4
day_of_year：1-365
week_of_year：1-52
is_weekend：0 or 1
month_sin/cos：Cyclical encoding for months
day_of_year_sin/cos：Cyclical encoding for seasons
```

### Holiday Features
```
is_holiday：1 if US holiday, 0 otherwise
days_to_holiday：Days until next holiday
days_since_holiday：Days since last holiday
```

### Feature Count: **28+ engineered features per record**

---

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip or conda
- Virtual environment (recommended)
- Sufficient disk space for models (~500MB)

### Step 1: Clone/Download Project

```bash
cd c:\Users\varen\OneDrive\Desktop\microgcc\forecasting-system
```

### Step 2: Create Virtual Environment

**Using venv:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

**Using conda:**
```bash
conda create -n forecasting python=3.10
conda activate forecasting
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** TensorFlow installation may take a few minutes.

### Step 4: Verify Installation

```bash
python -c "import pandas; import sklearn; import tensorflow; print('All imports successful!')"
```

---

## Installation

### Complete Installation Script

```bash
# 1. Navigate to project directory
cd forecasting-system

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verify installation
python -c "from src import preprocessing; print('Setup successful!')"
```

---

## Running the Pipeline

### Full Pipeline Execution

```bash
python run_pipeline.py
```

**What happens:**
1. ✅ Generates/loads sample data (if not exists)
2. ✅ Preprocesses data (handles missing values, outliers)
3. ✅ Engineers features (lags, rolling, dates, holidays)
4. ✅ Splits data temporally (train: 2022-2023, test: 2024)
5. ✅ Trains all 4 models per state
6. ✅ Evaluates models (MAE, RMSE, MAPE)
7. ✅ Selects best model by RMSE
8. ✅ Generates 8-week forecasts
9. ✅ Creates visualizations
10. ✅ Saves outputs to `outputs/` directory

**Execution Time:** 5-15 minutes (depending on data size)

**Output Files:**
- `outputs/forecasts/state_forecast.csv` - Forecast per state
- `outputs/plots/state_forecast.png` - Visualization per state
- `outputs/plots/model_comparison.png` - Overall comparison
- `outputs/metrics/state_metrics.json` - Evaluation metrics

---

## Running the API

### Start the API Server

```bash
# Option 1: Direct Python
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Option 2: Using the module
python api/app.py

# Option 3: From run_pipeline (if added)
python -c "from api.app import run_api; run_api()"
```

**Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

### Access API

- **API Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Base URL:** http://localhost:8000

---

## API Documentation

### Available Endpoints

#### 1. GET `/health` - Health Check
Returns API status and available models.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-05-07T10:30:00",
  "version": "1.0.0",
  "models_available": ["arima", "prophet", "xgboost", "lstm"]
}
```

---

#### 2. GET `/states` - List Available States
Returns list of states for which forecasts can be generated.

**Response:**
```json
{
  "states": ["California", "Florida", "Illinois", "New York", "Texas"],
  "total_states": 5
}
```

---

#### 3. GET `/forecast?state=California` - Get Forecast
Generates forecast for specified state.

**Query Parameters:**
- `state` (string, required): State name

**Response:**
```json
{
  "state": "California",
  "best_model": "xgboost",
  "forecast_horizon_days": 56,
  "forecast_horizon_weeks": 8,
  "forecast_start_date": "2026-05-08",
  "forecast_end_date": "2026-06-24",
  "total_predictions": 56,
  "predictions": [
    {
      "date": "2024-03-01",
      "forecast": 4523.45,
      "forecast_lower": 4297.28,
      "forecast_upper": 4749.62,
      "confidence": "95%"
    },
    ...
  ],
  "metadata": {
    "models_used": ["arima", "prophet", "xgboost", "lstm"],
    "data_points_in_training": 730,
    "last_historical_date": "2026-05-07",
    "api_version": "1.0.0",
    "generated_at": "2026-05-07T10:30:00"
  }
}
```

---

#### 4. GET `/batch-forecast` - Get All Forecasts
Generates forecasts for all available states in one request.

**Response:**
```json
{
  "total_states": 5,
  "successful": 5,
  "forecasts": {
    "California": {...},
    "Florida": {...},
    ...
  },
  "generated_at": "2026-05-07T10:30:00"
}
```

---

#### 5. GET `/model-comparison?state=California` - Model Metrics
Returns evaluation metrics for all models on a state.

**Response:**
```json
{
  "state": "California",
  "metrics": {
    "best_model": "xgboost",
    "best_rmse": 342.15,
    "best_mae": 278.90,
    "best_mape": 6.23,
    "ensemble_rmse": 385.42,
    "all_models": {
      "arima": {...},
      "prophet": {...},
      "xgboost": {...},
      "lstm": {...}
    }
  },
  "generated_at": "2024-03-01T10:30:00"
}
```

---

#### 6. GET `/data-info` - Data Statistics
Returns information about loaded data.

**Response:**
```json
{
  "total_records": 3650,
  "total_states": 5,
  "states": ["California", "Florida", "Illinois", "New York", "Texas"],
  "date_range": {
    "start": "2024-01-01",
    "end": "2026-05-07"
  },
  "total_features": 28,
  "features": ["date", "state", "sales", ...],
  "sample_size_per_state": {
    "California": 730,
    "Florida": 730,
    ...
  }
}
```

---

## Example API Calls

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# List states
curl http://localhost:8000/states

# Get forecast for California
curl "http://localhost:8000/forecast?state=California"

# Get batch forecast
curl http://localhost:8000/batch-forecast

# Model comparison
curl "http://localhost:8000/model-comparison?state=California"

# Data info
curl http://localhost:8000/data-info
```

### Using Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Get forecast
response = requests.get(f"{BASE_URL}/forecast", params={"state": "California"})
forecast = response.json()
print(f"Forecast for {forecast['state']}: {forecast['best_model']}")

# Get all forecasts
response = requests.get(f"{BASE_URL}/batch-forecast")
all_forecasts = response.json()
print(f"Generated forecasts for {all_forecasts['successful']} states")
```

### Using JavaScript/TypeScript

```javascript
// Fetch forecast
fetch('http://localhost:8000/forecast?state=California')
  .then(response => response.json())
  .then(data => {
    console.log(`Forecast for ${data.state}:`);
    console.log(`Best model: ${data.best_model}`);
    console.log(`${data.total_predictions} predictions`);
  });
```

---

## Evaluation Metrics

### Metrics Used

#### 1. **MAE (Mean Absolute Error)**
```
MAE = (1/n) * Σ|y_true - y_pred|
```
- **Interpretation:** Average absolute deviation
- **Unit:** Same as sales values
- **Range:** 0 to ∞ (lower is better)

#### 2. **RMSE (Root Mean Squared Error)**
```
RMSE = √[(1/n) * Σ(y_true - y_pred)²]
```
- **Interpretation:** Square root of mean squared error
- **Unit:** Same as sales values
- **Range:** 0 to ∞ (lower is better)
- **Note:** Penalizes large errors more

#### 3. **MAPE (Mean Absolute Percentage Error)**
```
MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|
```
- **Interpretation:** Percentage error relative to actual value
- **Unit:** Percentage (%)
- **Range:** 0 to ∞ (lower is better)
- **Advantage:** Scale-independent comparison

### Model Selection Criteria

**Primary:** RMSE on test set  
**Secondary:** MAE, MAPE for tie-breaking  
**Rationale:** RMSE penalizes larger errors more, important for business impact

### Expected Performance Ranges

| Metric | Good | Acceptable | Poor |
|--------|------|-----------|------|
| RMSE   | <300 | 300-500   | >500 |
| MAE    | <250 | 250-400   | >400 |
| MAPE   | <5%  | 5-10%     | >10% |

---

## Key Assumptions

### 1. **Data Assumptions**
- Sales data has daily granularity
- States are independent (separate models per state)
- Future patterns follow historical patterns
- No major structural breaks in data

### 2. **Temporal Assumptions**
- Data is ordered chronologically
- Sufficient historical data (minimum 100 days per state)
- Seasonality patterns consistent across years
- No extreme external shocks

### 3. **Forecasting Assumptions**
- 8-week forecast horizon is appropriate
- Holiday calendars are stable
- No new holidays or policy changes
- Business operations remain consistent

### 4. **Model Assumptions**
- ARIMA: Stationary or difference-stationary data
- Prophet: Time series with strong seasonality
- XGBoost: Lag features capture dependencies
- LSTM: Sufficient training data for deep learning

### 5. **Operational Assumptions**
- Models retrained monthly with latest data
- API error handling via circuit breaker pattern
- Monitoring and alerting in place
- Fallback to ensemble if best model fails

---

## Future Enhancements

### Short Term (1-2 months)
1. **Model Tuning:** Hyperparameter optimization via Bayesian search
2. **Exogenous Variables:** Include price, promotions, competitor data
3. **Transfer Learning:** Pre-trained models for new states
4. **Confidence Intervals:** Proper quantile forecasts via Prophet/ARIMA
5. **Model Retraining:** Automated monthly/weekly retraining pipeline

### Medium Term (2-6 months)
1. **Ensemble Methods:** Weighted ensemble based on individual model performance
2. **Anomaly Detection:** Detect and handle anomalies in forecasts
3. **Multiple Forecast Horizons:** 1-week, 4-week, 12-week ahead forecasts
4. **Database Integration:** Store predictions and historical performance in PostgreSQL
5. **Real-time API:** WebSocket support for streaming predictions

### Long Term (6+ months)
1. **Advanced Models:** Transformers (Attention-based), N-BEATS
2. **Multi-level Forecasting:** Regional aggregation, Top-down reconciliation
3. **Causal Inference:** Identify drivers of sales variation
4. **AutoML:** Auto-detect best models per state
5. **Cloud Deployment:** AWS/Azure/GCP containerization, scaling
6. **CI/CD Pipeline:** Automated testing, validation, deployment
7. **Model Monitoring:** Track model drift, retrain triggers
8. **Explainability:** SHAP values, feature importance analysis

---

## Support & Troubleshooting

### Common Issues

**1. TensorFlow Installation Failed**
```bash
pip install --upgrade tensorflow
# or
conda install tensorflow
```

**2. Insufficient Memory for LSTM**
- Reduce LSTM_BATCH_SIZE in config.py from 32 to 16
- Reduce LSTM_SEQUENCE_LENGTH from 30 to 20

**3. API Port Already in Use**
```bash
# Use different port
uvicorn api.app:app --port 8001
```

**4. Models Not Found Error**
- Run `python run_pipeline.py` first to train models
- Check `models/` directory contains *.pkl files

### Logging

All operations are logged to console. For file logging:

```python
# Add to config.py
LOG_TO_FILE = True
LOG_FILE = "forecasting.log"
```

### Contact

For issues or questions:
- **Email:** varenyavarma@example.com
- **GitHub:** [Link to repository]

---

## License

This project is confidential and intended for Microgcc internal use.

---

## Authors

**P. B. Varenya Varma**  
Data Science Case Study
UGID: 22WU0101018  
Role: Data Science Candidate

---

## Version History

| Version | Date       | Changes |
|---------|-----------|---------|
| 1.0.0   | 2026-05-07 | Initial release with 4 models, API, documentation |

---

**Last Updated:** 2026-05-07  
**Project Status:** Production-Ready

---

🚀 Quick Start:
# 1. Navigate to project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline (train models, forecast)
python run_pipeline.py

# 4. Start API
uvicorn api.app:app --reload

# 5. Access docs
http://localhost:8000/docs# 