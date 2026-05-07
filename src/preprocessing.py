"""
Data preprocessing module for time series data
Handles missing values, date alignment, and data cleaning
"""
import pandas as pd
import numpy as np
from src.config import (RAW_DATA_FILE, DATE_COLUMN, STATE_COLUMN, 
                         TARGET_COLUMN, TRAIN_TEST_SPLIT_DATE)
from src.utils import setup_logger

logger = setup_logger(__name__)


def load_data(filepath=None):
    """
    Load raw sales data from CSV
    
    Args:
        filepath: Path to CSV file (default: from config)
    
    Returns:
        DataFrame with columns: date, state, sales
    """
    if filepath is None:
        filepath = RAW_DATA_FILE
    
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    
    logger.info(f"Data loaded. Shape: {df.shape}")
    logger.info(f"States: {df[STATE_COLUMN].unique().tolist()}")
    logger.info(f"Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")
    
    return df


def handle_missing_dates(df, state):
    """
    Create complete date range and handle missing dates for a state
    
    Args:
        df: DataFrame for single state
        state: State name
    
    Returns:
        DataFrame with complete date range and forward-filled values
    """
    # Get date range
    min_date = df[DATE_COLUMN].min()
    max_date = df[DATE_COLUMN].max()
    
    # Create complete date range
    complete_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Reindex to complete date range
    df_complete = df.set_index(DATE_COLUMN).reindex(complete_dates)
    df_complete[STATE_COLUMN] = state
    df_complete = df_complete.reset_index()
    df_complete.rename(columns={'index': DATE_COLUMN}, inplace=True)
    
    logger.info(f"Handling missing dates for {state}: "
                f"Original {len(df)} -> Complete {len(df_complete)} records")
    
    return df_complete


def handle_missing_values(df, method='forward_fill', limit=7):
    """
    Handle missing values in sales data
    
    Args:
        df: DataFrame
        method: 'forward_fill', 'interpolate', or 'drop'
        limit: Maximum consecutive periods to fill
    
    Returns:
        DataFrame with missing values handled
    """
    logger.info(f"Handling missing values using {method} method")
    
    missing_before = df[TARGET_COLUMN].isnull().sum()
    
    if method == 'forward_fill':
        df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(method='ffill', limit=limit)
        df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(method='bfill', limit=limit)
    
    elif method == 'interpolate':
        df[TARGET_COLUMN] = df[TARGET_COLUMN].interpolate(method='linear', limit_area='inside')
        df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(method='bfill')
    
    elif method == 'drop':
        df = df.dropna(subset=[TARGET_COLUMN])
    
    missing_after = df[TARGET_COLUMN].isnull().sum()
    logger.info(f"Missing values: {missing_before} -> {missing_after}")
    
    return df.reset_index(drop=True)


def sort_data(df):
    """
    Sort data by state and date
    
    Args:
        df: DataFrame
    
    Returns:
        Sorted DataFrame
    """
    df = df.sort_values([STATE_COLUMN, DATE_COLUMN]).reset_index(drop=True)
    logger.info("Data sorted by state and date")
    return df


def remove_outliers(df, state, lower_percentile=1, upper_percentile=99):
    """
    Remove outliers using percentile method
    
    Args:
        df: DataFrame for single state
        state: State name
        lower_percentile: Lower percentile threshold
        upper_percentile: Upper percentile threshold
    
    Returns:
        DataFrame with outliers handled
    """
    lower_bound = df[TARGET_COLUMN].quantile(lower_percentile / 100)
    upper_bound = df[TARGET_COLUMN].quantile(upper_percentile / 100)
    
    outliers = ((df[TARGET_COLUMN] < lower_bound) | (df[TARGET_COLUMN] > upper_bound)).sum()
    
    # Cap outliers instead of removing
    df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(lower=lower_bound, upper=upper_bound)
    
    logger.info(f"{state}: Handled {outliers} outliers "
                f"(bounds: {lower_bound:.2f} - {upper_bound:.2f})")
    
    return df


def preprocess_pipeline(filepath=None):
    """
    Complete preprocessing pipeline
    
    Args:
        filepath: Path to raw data
    
    Returns:
        Processed DataFrame ready for feature engineering
    """
    logger.info("=" * 60)
    logger.info("Starting Preprocessing Pipeline")
    logger.info("=" * 60)
    
    # Load data
    df = load_data(filepath)
    
    # Sort data
    df = sort_data(df)
    
    # Process each state
    processed_states = []
    for state in df[STATE_COLUMN].unique():
        logger.info(f"\nProcessing state: {state}")
        
        state_df = df[df[STATE_COLUMN] == state].copy()
        
        # Handle missing dates
        state_df = handle_missing_dates(state_df, state)
        
        # Handle missing values
        state_df = handle_missing_values(state_df, method='interpolate', limit=7)
        
        # Remove outliers
        state_df = remove_outliers(state_df, state)
        
        processed_states.append(state_df)
    
    # Combine all states
    df_processed = pd.concat(processed_states, ignore_index=True)
    df_processed = sort_data(df_processed)
    
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing Complete")
    logger.info(f"Final shape: {df_processed.shape}")
    logger.info(f"Date range: {df_processed[DATE_COLUMN].min()} to {df_processed[DATE_COLUMN].max()}")
    logger.info(f"Total missing values: {df_processed.isnull().sum().sum()}")
    logger.info("=" * 60)
    
    return df_processed


def split_train_test(df, split_date=TRAIN_TEST_SPLIT_DATE, state=None):
    """
    Split data into train and test sets using time-based split (NO random split)
    
    Args:
        df: DataFrame
        split_date: Date to split at
        state: Filter by state (optional)
    
    Returns:
        Tuple of (train_df, test_df)
    """
    if state is not None:
        df = df[df[STATE_COLUMN] == state].copy()
    
    split_date = pd.to_datetime(split_date)
    
    train_df = df[df[DATE_COLUMN] < split_date].copy()
    test_df = df[df[DATE_COLUMN] >= split_date].copy()
    
    logger.info(f"Train-Test Split (temporal):")
    logger.info(f"  Split date: {split_date}")
    logger.info(f"  Train: {len(train_df)} samples ({train_df[DATE_COLUMN].min()} to {train_df[DATE_COLUMN].max()})")
    logger.info(f"  Test: {len(test_df)} samples ({test_df[DATE_COLUMN].min()} to {test_df[DATE_COLUMN].max()})")
    
    return train_df, test_df


def get_state_data(df, state):
    """
    Get data for a specific state
    
    Args:
        df: DataFrame
        state: State name
    
    Returns:
        DataFrame for the state
    """
    return df[df[STATE_COLUMN] == state].copy().reset_index(drop=True)


def data_quality_report(df):
    """
    Generate data quality report
    
    Args:
        df: DataFrame
    
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'total_records': len(df),
        'date_range': f"{df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}",
        'states': df[STATE_COLUMN].nunique(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'sales_stats': {
            'mean': df[TARGET_COLUMN].mean(),
            'std': df[TARGET_COLUMN].std(),
            'min': df[TARGET_COLUMN].min(),
            'max': df[TARGET_COLUMN].max(),
        }
    }
    
    logger.info("\nData Quality Report:")
    logger.info(f"  Total records: {report['total_records']}")
    logger.info(f"  Date range: {report['date_range']}")
    logger.info(f"  Number of states: {report['states']}")
    logger.info(f"  Duplicates: {report['duplicates']}")
    
    return report
