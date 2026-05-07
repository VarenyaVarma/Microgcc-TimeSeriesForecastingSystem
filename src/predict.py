"""
Prediction module for generating forecasts
Uses trained models to forecast future sales
"""
import pandas as pd
import numpy as np
from src.utils import setup_logger, load_model
from src.config import FORECAST_HORIZON, DATE_COLUMN, STATE_COLUMN, TARGET_COLUMN
from src.train_models import forecast_arima, forecast_prophet, forecast_xgboost, forecast_lstm
from src.feature_engineering import get_feature_columns

logger = setup_logger(__name__)


def generate_future_dates(last_date, periods=FORECAST_HORIZON):
    """
    Generate future dates for forecasting
    
    Args:
        last_date: Last date in training data
        periods: Number of periods to forecast
    
    Returns:
        DatetimeIndex with future dates
    """
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
    return future_dates


def create_future_features(future_dates, last_row):
    """
    Create features for future dates
    
    Args:
        future_dates: DatetimeIndex for future
        last_row: Last row from training data (for lag features, etc.)
    
    Returns:
        DataFrame with features for future periods
    """
    future_df = pd.DataFrame({DATE_COLUMN: future_dates})
    
    # Date features
    future_df['day_of_week'] = future_df[DATE_COLUMN].dt.dayofweek.astype('int8')
    future_df['day_of_month'] = future_df[DATE_COLUMN].dt.day.astype('int8')
    future_df['month'] = future_df[DATE_COLUMN].dt.month.astype('int8')
    future_df['quarter'] = future_df[DATE_COLUMN].dt.quarter.astype('int8')
    future_df['day_of_year'] = future_df[DATE_COLUMN].dt.dayofyear.astype('int16')
    future_df['week_of_year'] = future_df[DATE_COLUMN].dt.isocalendar().week.astype('int8')
    future_df['is_weekend'] = (future_df[DATE_COLUMN].dt.dayofweek >= 5).astype('int8')
    
    # Cyclical encoding
    future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
    future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
    future_df['day_of_year_sin'] = np.sin(2 * np.pi * future_df['day_of_year'] / 365)
    future_df['day_of_year_cos'] = np.cos(2 * np.pi * future_df['day_of_year'] / 365)
    
    # Holiday features (simplified - set to 0 for future)
    future_df['is_holiday'] = 0
    future_df['days_to_holiday'] = 30
    future_df['days_since_holiday'] = 30
    
    # Lag features (use last row values)
    lag_cols = [col for col in last_row.index if 'lag_' in col]
    for col in lag_cols:
        future_df[col] = last_row[col] if col in last_row.index else 0
    
    # Rolling features (use last row values)
    rolling_cols = [col for col in last_row.index if 'rolling_' in col]
    for col in rolling_cols:
        future_df[col] = last_row[col] if col in last_row.index else 0
    
    return future_df


def predict_with_arima(state, models_dict):
    """
    Generate ARIMA prediction
    
    Args:
        state: State name
        models_dict: Dictionary of trained models
    
    Returns:
        Array of predictions
    """
    try:
        model = models_dict.get('arima')
        if model is None:
            model = load_model('arima', state)
        
        if model is None:
            logger.warning(f"ARIMA model not found for {state}")
            return None
        
        predictions = forecast_arima(model, steps=FORECAST_HORIZON)
        logger.info(f"Generated ARIMA forecast for {state}: {len(predictions)} periods")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error generating ARIMA forecast for {state}: {str(e)}")
        return None


def predict_with_prophet(state, models_dict):
    """
    Generate Prophet prediction
    
    Args:
        state: State name
        models_dict: Dictionary of trained models
    
    Returns:
        Array of predictions
    """
    try:
        model = models_dict.get('prophet')
        if model is None:
            model = load_model('prophet', state)
        
        if model is None:
            logger.warning(f"Prophet model not found for {state}")
            return None
        
        predictions = forecast_prophet(model, steps=FORECAST_HORIZON)
        logger.info(f"Generated Prophet forecast for {state}: {len(predictions)} periods")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error generating Prophet forecast for {state}: {str(e)}")
        return None


def predict_with_xgboost(state, df_full, models_dict):
    """
    Generate XGBoost prediction
    
    Args:
        state: State name
        df_full: Full data with features
        models_dict: Dictionary of trained models
    
    Returns:
        Array of predictions
    """
    try:
        model = models_dict.get('xgboost')
        if model is None:
            model = load_model('xgboost', state)
        
        if model is None:
            logger.warning(f"XGBoost model not found for {state}")
            return None
        
        # Prepare future features
        state_df = df_full[df_full[STATE_COLUMN] == state].copy()
        last_date = state_df[DATE_COLUMN].max()
        last_row = state_df.iloc[-1]
        
        future_dates = generate_future_dates(last_date, FORECAST_HORIZON)
        future_df = create_future_features(future_dates, last_row)
        
        predictions = forecast_xgboost(model, future_df, steps=FORECAST_HORIZON)
        logger.info(f"Generated XGBoost forecast for {state}: {len(predictions)} periods")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error generating XGBoost forecast for {state}: {str(e)}")
        return None


def predict_with_lstm(state, df_full, models_dict):
    """
    Generate LSTM prediction
    
    Args:
        state: State name
        df_full: Full data
        models_dict: Dictionary of trained models
    
    Returns:
        Array of predictions
    """
    try:
        model = models_dict.get('lstm')
        if model is None:
            model = load_model('lstm', state)
        
        if model is None:
            logger.warning(f"LSTM model not found for {state}")
            return None
        
        # Prepare data
        state_df = df_full[df_full[STATE_COLUMN] == state].copy()
        sales_data = state_df[TARGET_COLUMN].values
        
        # Get scaler
        scaler = model.scaler
        if scaler is None:
            scaler = load_model(f'lstm_{state}_scaler', 'scaler')
        
        # Scale data
        sales_scaled = scaler.transform(sales_data.reshape(-1, 1))
        
        predictions = forecast_lstm(model, sales_scaled, steps=FORECAST_HORIZON)
        logger.info(f"Generated LSTM forecast for {state}: {len(predictions)} periods")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error generating LSTM forecast for {state}: {str(e)}")
        return None


def generate_all_predictions(state, df_full, models_dict, best_model=None):
    """
    Generate predictions from all models
    
    Args:
        state: State name
        df_full: Full data with features
        models_dict: Dictionary of trained models
        best_model: Name of best model to use (optional)
    
    Returns:
        Dictionary with predictions from all models
    """
    logger.info(f"\nGenerating predictions for {state}")
    logger.info("-" * 40)
    
    predictions = {}
    
    # Get predictions from each model
    arima_pred = predict_with_arima(state, models_dict)
    if arima_pred is not None:
        predictions['arima'] = arima_pred
    
    prophet_pred = predict_with_prophet(state, models_dict)
    if prophet_pred is not None:
        predictions['prophet'] = prophet_pred
    
    xgboost_pred = predict_with_xgboost(state, df_full, models_dict)
    if xgboost_pred is not None:
        predictions['xgboost'] = xgboost_pred
    
    lstm_pred = predict_with_lstm(state, df_full, models_dict)
    if lstm_pred is not None:
        predictions['lstm'] = lstm_pred
    
    # Use best model if specified
    if best_model and best_model in predictions:
        final_prediction = predictions[best_model]
        logger.info(f"Using {best_model} for final forecast")
    else:
        # Ensemble: Average of all predictions
        valid_preds = [p for p in predictions.values() if p is not None]
        if valid_preds:
            final_prediction = np.mean(valid_preds, axis=0)
            logger.info(f"Using ensemble forecast (average of {len(valid_preds)} models)")
        else:
            logger.error(f"No valid predictions for {state}")
            final_prediction = None
    
    return {
        'state': state,
        'all_predictions': predictions,
        'final_prediction': final_prediction,
        'best_model': best_model
    }


def create_forecast_dataframe(forecast_results):
    """
    Create properly formatted forecast dataframe
    
    Args:
        forecast_results: Forecast results from generate_all_predictions
    
    Returns:
        DataFrame with forecast
    """
    state = forecast_results['state']
    final_prediction = forecast_results['final_prediction']
    
    # Get last date from data
    logger.info(f"Creating forecast dataframe for {state}")
    
    # Generate future dates  
    # We need the last date - this will be provided by the caller
    from datetime import datetime
    today = datetime(2026, 5, 7)
    forecast_dates = pd.date_range(start=today + pd.Timedelta(days=1), periods=FORECAST_HORIZON, freq='D')
    
    forecast_df = pd.DataFrame({
        DATE_COLUMN: forecast_dates,
        STATE_COLUMN: state,
        'forecast': final_prediction,
        'forecast_lower': final_prediction * 0.95,
        'forecast_upper': final_prediction * 1.05
    })
    
    return forecast_df
