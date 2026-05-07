"""
Configuration settings for the forecasting system
"""
import os
from datetime import datetime

# Project Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
FORECASTS_DIR = os.path.join(OUTPUTS_DIR, 'forecasts')
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')
METRICS_DIR = os.path.join(OUTPUTS_DIR, 'metrics')

# Data Configuration
RAW_DATA_FILE = os.path.join(DATA_DIR, 'sales_data.csv')
DATE_COLUMN = 'date'
STATE_COLUMN = 'state'
TARGET_COLUMN = 'sales'

# Time Series Configuration
FORECAST_HORIZON = 56  # 8 weeks (7 days * 8)
TRAIN_TEST_SPLIT_DATE = '2026-04-01'  # Date to split train/test
VALIDATION_SIZE = 30  # Days for validation set

# Model Configuration
MODELS_TO_TRAIN = ['arima', 'prophet', 'xgboost', 'lstm']

# Feature Engineering
LAG_FEATURES = [1, 7, 30]  # t-1, t-7, t-30
ROLLING_WINDOW_SIZES = [7, 30]  # 7-day and 30-day rolling stats

# ARIMA Parameters
ARIMA_ORDER = (1, 1, 1)  # (p, d, q) - can be tuned per state
ARIMA_SEASONAL_ORDER = (1, 1, 1, 7)  # (P, D, Q, s) - 7-day seasonality

# Prophet Configuration
PROPHET_YEARLY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = True
PROPHET_INTERVAL_WIDTH = 0.95

# XGBoost Configuration
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 7,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'early_stopping_rounds': 50,
}

# LSTM Configuration
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32
LSTM_SEQUENCE_LENGTH = 30
LSTM_UNITS = 128
LSTM_DROPOUT = 0.2
LSTM_VALIDATION_SPLIT = 0.2
LSTM_EARLY_STOPPING_PATIENCE = 10

# API Configuration
API_HOST = '0.0.0.0'
API_PORT = 8000
API_RELOAD = True

# Evaluation Metrics
METRICS = ['mae', 'rmse', 'mape']

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Random Seed
RANDOM_SEED = 42

# Dates for holidays
COUNTRIES = ['US']  # Use US holidays

# Create necessary directories
for directory in [MODELS_DIR, FORECASTS_DIR, PLOTS_DIR, METRICS_DIR]:
    os.makedirs(directory, exist_ok=True)
