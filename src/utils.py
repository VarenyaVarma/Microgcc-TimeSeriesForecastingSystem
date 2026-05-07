"""
Utility functions for the forecasting system
"""
import logging
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from src.config import LOG_FORMAT, LOG_LEVEL, MODELS_DIR, METRICS_DIR

# Setup logging
def setup_logger(name):
    """
    Setup logger for the module
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    
    return logger


def save_model(model, model_name, state):
    """
    Save trained model to disk
    
    Args:
        model: Trained model object
        model_name: Name of the model (e.g., 'arima', 'prophet')
        state: State name
    """
    filename = f'{state}_{model_name}_model.pkl'
    filepath = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, filepath)
    logger = setup_logger(__name__)
    logger.info(f"Model saved: {filepath}")
    return filepath


def load_model(model_name, state):
    """
    Load trained model from disk
    
    Args:
        model_name: Name of the model
        state: State name
    
    Returns:
        Loaded model object
    """
    filename = f'{state}_{model_name}_model.pkl'
    filepath = os.path.join(MODELS_DIR, filename)
    
    if not os.path.exists(filepath):
        return None
    
    return joblib.load(filepath)


def save_metrics(metrics_dict, state):
    """
    Save evaluation metrics to JSON
    
    Args:
        metrics_dict: Dictionary of metrics
        state: State name
    """
    filename = f'{state}_metrics.json'
    filepath = os.path.join(METRICS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    logger = setup_logger(__name__)
    logger.info(f"Metrics saved: {filepath}")
    return filepath


def load_metrics(state):
    """
    Load evaluation metrics from JSON
    
    Args:
        state: State name
    
    Returns:
        Dictionary of metrics
    """
    filename = f'{state}_metrics.json'
    filepath = os.path.join(METRICS_DIR, filename)
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        MAPE value
    """
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def create_timestamp():
    """Create a timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_test_data_csv():
    """
    Create sample sales data CSV for testing
    """
    from src.config import RAW_DATA_FILE, DATA_DIR
    
    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2026-05-07', freq='D')
    states = ['California', 'Texas', 'Florida', 'New York', 'Illinois']
    
    data = []
    for state in states:
        base_sales = np.random.uniform(2000, 5000)
        for i, date in enumerate(dates):
            # Trend
            trend = base_sales + (i * np.random.uniform(0.5, 1.5))
            # Seasonality
            seasonal = 1000 * np.sin(2 * np.pi * date.dayofyear / 365)
            # Weekly pattern
            weekly = 200 if date.dayofweek >= 4 else 0
            # Noise
            noise = np.random.normal(0, 300)
            
            sales = trend + seasonal + weekly + noise
            sales = max(sales, 100)
            
            # 5% missing values
            if np.random.random() > 0.95:
                sales = np.nan
            
            data.append({
                'date': date,
                'state': state,
                'sales': sales
            })
    
    df = pd.DataFrame(data)
    
    # Remove some dates to simulate gaps
    mask = (df['date'].dt.dayofweek == 2) & (df['state'] == 'Texas') & (df['date'].dt.month == 6)
    df = df[~mask].reset_index(drop=True)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(RAW_DATA_FILE, index=False)
    
    logger = setup_logger(__name__)
    logger.info(f"Test data created: {RAW_DATA_FILE}")
    logger.info(f"Shape: {df.shape}, Missing values: {df.isnull().sum().sum()}")
    
    return df
