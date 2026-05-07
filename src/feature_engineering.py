"""
Feature engineering module for time series forecasting
Creates lag features, rolling features, date features, and holiday features
"""
import pandas as pd
import numpy as np
from src.config import (LAG_FEATURES, ROLLING_WINDOW_SIZES, DATE_COLUMN, 
                        STATE_COLUMN, TARGET_COLUMN, COUNTRIES)
from src.utils import setup_logger
import holidays

logger = setup_logger(__name__)


def create_lag_features(df, lags=None):
    """
    Create lagged features for sales
    
    Args:
        df: DataFrame with sales data
        lags: List of lag values (default: from config)
    
    Returns:
        DataFrame with lag features added
    """
    if lags is None:
        lags = LAG_FEATURES
    
    df = df.copy()
    logger.info(f"Creating lag features: {lags}")
    
    for lag in lags:
        col_name = f'{TARGET_COLUMN}_lag_{lag}'
        df[col_name] = df.groupby(STATE_COLUMN)[TARGET_COLUMN].shift(lag)
    
    logger.info(f"Added {len(lags)} lag features")
    return df


def create_rolling_features(df, windows=None):
    """
    Create rolling window features (mean and std)
    
    Args:
        df: DataFrame with sales data
        windows: List of window sizes (default: from config)
    
    Returns:
        DataFrame with rolling features added
    """
    if windows is None:
        windows = ROLLING_WINDOW_SIZES
    
    df = df.copy()
    logger.info(f"Creating rolling features: {windows}")
    
    for window in windows:
        # Rolling mean
        col_name = f'{TARGET_COLUMN}_rolling_mean_{window}'
        df[col_name] = df.groupby(STATE_COLUMN)[TARGET_COLUMN].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling std
        col_name = f'{TARGET_COLUMN}_rolling_std_{window}'
        df[col_name] = df.groupby(STATE_COLUMN)[TARGET_COLUMN].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    logger.info(f"Added {len(windows) * 2} rolling features")
    return df


def create_date_features(df):
    """
    Create temporal features from date
    
    Args:
        df: DataFrame with date column
    
    Returns:
        DataFrame with date features added
    """
    df = df.copy()
    logger.info("Creating date features")
    
    df['day_of_week'] = df[DATE_COLUMN].dt.dayofweek
    df['day_of_week'] = df['day_of_week'].astype('int8')
    
    df['day_of_month'] = df[DATE_COLUMN].dt.day
    df['day_of_month'] = df['day_of_month'].astype('int8')
    
    df['month'] = df[DATE_COLUMN].dt.month
    df['month'] = df['month'].astype('int8')
    
    df['quarter'] = df[DATE_COLUMN].dt.quarter
    df['quarter'] = df['quarter'].astype('int8')
    
    df['day_of_year'] = df[DATE_COLUMN].dt.dayofyear
    df['day_of_year'] = df['day_of_year'].astype('int16')
    
    df['week_of_year'] = df[DATE_COLUMN].dt.isocalendar().week
    df['week_of_year'] = df['week_of_year'].astype('int8')
    
    df['is_weekend'] = (df[DATE_COLUMN].dt.dayofweek >= 5).astype('int8')
    
    # Sine and cosine features for month (cyclical encoding)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Sine and cosine features for day of year
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    logger.info("Added 11 date features")
    return df


def create_holiday_features(df):
    """
    Add holiday flag features for US holidays
    
    Args:
        df: DataFrame with date column
    
    Returns:
        DataFrame with holiday features added
    """
    df = df.copy()
    logger.info("Creating holiday features")
    
    # Get US holidays
    us_holidays = holidays.US()
    
    df['is_holiday'] = df[DATE_COLUMN].dt.date.apply(
        lambda x: 1 if x in us_holidays else 0
    ).astype('int8')
    
    # Days to next holiday
    df['days_to_holiday'] = df[DATE_COLUMN].apply(
        lambda x: _days_to_next_holiday(x, us_holidays)
    ).astype('int16')
    
    # Days since last holiday
    df['days_since_holiday'] = df[DATE_COLUMN].apply(
        lambda x: _days_since_last_holiday(x, us_holidays)
    ).astype('int16')
    
    logger.info("Added 3 holiday features")
    return df


def _days_to_next_holiday(date, holiday_dict, max_days=365):
    """Calculate days to next holiday"""
    for i in range(1, max_days + 1):
        future_date = date + pd.Timedelta(days=i)
        if future_date.date() in holiday_dict:
            return i
    return max_days


def _days_since_last_holiday(date, holiday_dict, max_days=365):
    """Calculate days since last holiday"""
    for i in range(1, max_days + 1):
        past_date = date - pd.Timedelta(days=i)
        if past_date.date() in holiday_dict:
            return i
    return max_days


def fill_feature_na_values(df):
    """
    Fill NA values in engineered features
    
    Args:
        df: DataFrame with features
    
    Returns:
        DataFrame with NAs filled
    """
    df = df.copy()
    
    # Fill lag features with forward/backward fill per state
    lag_cols = [col for col in df.columns if 'lag_' in col]
    for col in lag_cols:
        df[col] = df.groupby(STATE_COLUMN)[col].transform(
            lambda x: x.fillna(method='bfill').fillna(method='ffill')
        )
    
    # Fill rolling features
    rolling_cols = [col for col in df.columns if 'rolling_' in col]
    for col in rolling_cols:
        df[col] = df.groupby(STATE_COLUMN)[col].transform(
            lambda x: x.fillna(method='bfill').fillna(x.mean())
        )
    
    # Fill any remaining NaN values with mean
    df = df.fillna(df.mean())
    
    logger.info(f"Filled NA values. Remaining NAs: {df.isnull().sum().sum()}")
    return df


def feature_engineering_pipeline(df):
    """
    Complete feature engineering pipeline
    
    Args:
        df: Preprocessed DataFrame
    
    Returns:
        DataFrame with all engineered features
    """
    logger.info("=" * 60)
    logger.info("Starting Feature Engineering Pipeline")
    logger.info("=" * 60)
    
    # Create features
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_date_features(df)
    df = create_holiday_features(df)
    
    # Fill NA values
    df = fill_feature_na_values(df)
    
    logger.info("\n" + "=" * 60)
    logger.info("Feature Engineering Complete")
    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Total features: {len(df.columns) - 3}")  # Exclude date, state, sales
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info("=" * 60)
    
    return df


def get_feature_columns(df):
    """
    Get list of feature columns (excluding date, state, target)
    
    Args:
        df: DataFrame
    
    Returns:
        List of feature columns
    """
    exclude_cols = [DATE_COLUMN, STATE_COLUMN, TARGET_COLUMN]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def scale_features(df, scaler=None, fit=True, feature_cols=None):
    """
    Scale features for neural networks
    
    Args:
        df: DataFrame
        scaler: StandardScaler object (optional)
        fit: Whether to fit the scaler
        feature_cols: Columns to scale (optional)
    
    Returns:
        Scaled DataFrame, scaler object
    """
    from sklearn.preprocessing import StandardScaler
    
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    
    if scaler is None:
        scaler = StandardScaler()
    
    df = df.copy()
    
    if fit:
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df, scaler
