"""
Model training module for time series forecasting
Implements ARIMA/SARIMA, Prophet, XGBoost, and LSTM models
"""
import pandas as pd
import numpy as np
from src.utils import setup_logger, save_model
from src.config import (ARIMA_ORDER, ARIMA_SEASONAL_ORDER, FORECAST_HORIZON,
                        TARGET_COLUMN, DATE_COLUMN, STATE_COLUMN, 
                        PROPHET_YEARLY_SEASONALITY, PROPHET_WEEKLY_SEASONALITY,
                        XGBOOST_PARAMS, LSTM_EPOCHS, LSTM_BATCH_SIZE, 
                        LSTM_SEQUENCE_LENGTH, LSTM_UNITS, LSTM_DROPOUT,
                        LSTM_VALIDATION_SPLIT, LSTM_EARLY_STOPPING_PATIENCE)
from src.feature_engineering import get_feature_columns

logger = setup_logger(__name__)


# ============================================================================
# ARIMA/SARIMA Model
# ============================================================================

def train_arima_model(df, state, order=None, seasonal_order=None):
    """
    Train ARIMA/SARIMA model
    
    Args:
        df: Training data for a single state
        state: State name
        order: ARIMA order tuple (p, d, q)
        seasonal_order: Seasonal ARIMA order tuple (P, D, Q, s)
    
    Returns:
        Trained ARIMA model
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    if order is None:
        order = ARIMA_ORDER
    if seasonal_order is None:
        seasonal_order = ARIMA_SEASONAL_ORDER
    
    try:
        logger.info(f"Training ARIMA for {state}: order={order}, seasonal={seasonal_order}")
        
        model = SARIMAX(
            df[TARGET_COLUMN],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        result = model.fit(disp=False, maxiter=1000)
        
        logger.info(f"ARIMA trained successfully for {state}")
        save_model(result, 'arima', state)
        
        return result
    
    except Exception as e:
        logger.error(f"Error training ARIMA for {state}: {str(e)}")
        return None


def forecast_arima(model, steps=FORECAST_HORIZON):
    """
    Generate ARIMA forecast
    
    Args:
        model: Trained ARIMA model
        steps: Number of steps to forecast
    
    Returns:
        Forecast values
    """
    try:
        forecast = model.get_forecast(steps=steps)
        predictions = forecast.predicted_mean.values
        return predictions
    except Exception as e:
        logger.error(f"Error forecasting with ARIMA: {str(e)}")
        return None


# ============================================================================
# Prophet Model
# ============================================================================

def train_prophet_model(df, state):
    """
    Train Facebook Prophet model
    
    Args:
        df: Training data for a single state
        state: State name
    
    Returns:
        Trained Prophet model
    """
    from prophet import Prophet
    
    try:
        logger.info(f"Training Prophet for {state}")
        
        # Prepare data in Prophet format
        prophet_df = df[[DATE_COLUMN, TARGET_COLUMN]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Remove NaN values
        prophet_df = prophet_df.dropna()
        
        # Create and train model
        model = Prophet(
            yearly_seasonality=PROPHET_YEARLY_SEASONALITY,
            weekly_seasonality=PROPHET_WEEKLY_SEASONALITY,
            daily_seasonality=False,
            interval_width=0.95
        )
        
        model.fit(prophet_df)
        
        logger.info(f"Prophet trained successfully for {state}")
        save_model(model, 'prophet', state)
        
        return model
    
    except Exception as e:
        logger.error(f"Error training Prophet for {state}: {str(e)}")
        return None


def forecast_prophet(model, steps=FORECAST_HORIZON):
    """
    Generate Prophet forecast
    
    Args:
        model: Trained Prophet model
        steps: Number of steps to forecast
    
    Returns:
        Forecast values
    """
    try:
        future = model.make_future_dataframe(periods=steps, freq='D')
        forecast = model.predict(future)
        predictions = forecast['yhat'][-steps:].values
        return predictions
    except Exception as e:
        logger.error(f"Error forecasting with Prophet: {str(e)}")
        return None


# ============================================================================
# XGBoost Model
# ============================================================================

def train_xgboost_model(df_train, df_val, state):
    """
    Train XGBoost model with lag features
    
    Args:
        df_train: Training data
        df_val: Validation data
        state: State name
    
    Returns:
        Trained XGBoost model
    """
    import xgboost as xgb
    
    try:
        logger.info(f"Training XGBoost for {state}")
        
        feature_cols = get_feature_columns(df_train)
        
        X_train = df_train[feature_cols].fillna(0)
        y_train = df_train[TARGET_COLUMN]
        
        X_val = df_val[feature_cols].fillna(0)
        y_val = df_val[TARGET_COLUMN]
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model
        params = XGBOOST_PARAMS.copy()
        
        evals = [(dtrain, 'train'), (dval, 'eval')]
        evals_result = {}
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        logger.info(f"XGBoost trained successfully for {state}")
        save_model(model, 'xgboost', state)
        
        # Store features for later use
        model.feature_names = feature_cols
        
        return model
    
    except Exception as e:
        logger.error(f"Error training XGBoost for {state}: {str(e)}")
        return None


def forecast_xgboost(model, df_forecast, steps=FORECAST_HORIZON):
    """
    Generate XGBoost forecast
    
    Args:
        model: Trained XGBoost model
        df_forecast: DataFrame with features for forecasting
        steps: Number of steps to forecast
    
    Returns:
        Forecast values
    """
    import xgboost as xgb
    
    try:
        feature_cols = model.feature_names
        
        X_forecast = df_forecast[feature_cols].fillna(0)
        
        dforecast = xgb.DMatrix(X_forecast)
        predictions = model.predict(dforecast)
        
        return predictions[:steps]
    
    except Exception as e:
        logger.error(f"Error forecasting with XGBoost: {str(e)}")
        return None


# ============================================================================
# LSTM Model
# ============================================================================

def train_lstm_model(df_train, df_val, state):
    """
    Train LSTM neural network model
    
    Args:
        df_train: Training data
        df_val: Validation data
        state: State name
    
    Returns:
        Trained LSTM model
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import StandardScaler
    
    try:
        logger.info(f"Training LSTM for {state}")
        
        # Prepare data
        y_train = df_train[TARGET_COLUMN].values.reshape(-1, 1)
        y_val = df_val[TARGET_COLUMN].values.reshape(-1, 1)
        
        # Scale data
        scaler = StandardScaler()
        y_train_scaled = scaler.fit_transform(y_train)
        y_val_scaled = scaler.transform(y_val)
        
        # Create sequences
        seq_length = LSTM_SEQUENCE_LENGTH
        X_train, y_train_seq = _create_sequences(y_train_scaled, seq_length)
        X_val, y_val_seq = _create_sequences(y_val_scaled, seq_length)
        
        # Build model
        model = Sequential([
            LSTM(LSTM_UNITS, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
            Dropout(LSTM_DROPOUT),
            LSTM(LSTM_UNITS//2, activation='relu'),
            Dropout(LSTM_DROPOUT),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=LSTM_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        
        # Train model
        model.fit(
            X_train, y_train_seq,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            validation_data=(X_val, y_val_seq),
            callbacks=[early_stop],
            verbose=0
        )
        
        logger.info(f"LSTM trained successfully for {state}")
        
        # Save model and scaler
        save_model(model, 'lstm', state)
        save_model(scaler, f'lstm_{state}_scaler', 'scaler')
        
        # Store sequence length
        model.sequence_length = seq_length
        model.scaler = scaler
        
        return model
    
    except Exception as e:
        logger.error(f"Error training LSTM for {state}: {str(e)}")
        return None


def forecast_lstm(model, last_sequence, steps=FORECAST_HORIZON):
    """
    Generate LSTM forecast
    
    Args:
        model: Trained LSTM model
        last_sequence: Last sequence from training data
        steps: Number of steps to forecast
    
    Returns:
        Forecast values
    """
    try:
        seq_length = model.sequence_length
        scaler = model.scaler
        
        # Initialize with last sequence
        current_sequence = last_sequence[-seq_length:].reshape(1, seq_length, 1)
        predictions = []
        
        for _ in range(steps):
            # Predict next value
            next_pred = model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.append(current_sequence[:, 1:, :], 
                                        next_pred.reshape(1, 1, 1), axis=1)
        
        # Inverse scale predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    except Exception as e:
        logger.error(f"Error forecasting with LSTM: {str(e)}")
        return None


def _create_sequences(data, seq_length):
    """
    Create sequences for LSTM training
    
    Args:
        data: Input data
        seq_length: Sequence length
    
    Returns:
        Tuple of (X, y) sequences
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    
    return np.array(X), np.array(y)


# ============================================================================
# Main Training Pipeline
# ============================================================================

def train_all_models(df_train, df_val, state):
    """
    Train all models for a state
    
    Args:
        df_train: Training data
        df_val: Validation data
        state: State name
    
    Returns:
        Dictionary with all trained models
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"Training all models for {state}")
    logger.info("=" * 60)
    
    models = {}
    
    # ARIMA
    arima_model = train_arima_model(df_train, state)
    models['arima'] = arima_model
    
    # Prophet
    prophet_model = train_prophet_model(df_train, state)
    models['prophet'] = prophet_model
    
    # XGBoost
    xgboost_model = train_xgboost_model(df_train, df_val, state)
    models['xgboost'] = xgboost_model
    
    # LSTM
    lstm_model = train_lstm_model(df_train, df_val, state)
    models['lstm'] = lstm_model
    
    logger.info("\n" + "=" * 60)
    logger.info(f"All models trained for {state}")
    logger.info("=" * 60)
    
    return models
