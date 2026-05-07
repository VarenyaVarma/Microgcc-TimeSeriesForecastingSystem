"""
Generate sample sales data for time series forecasting
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sales_data(start_date='2024-01-01', end_date='2026-05-07', states=None):
    """
    Generate synthetic sales data with realistic patterns
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        states: List of states (default: Indian states)
    
    Returns:
        DataFrame with columns: date, state, sales
    """
    if states is None:
        states = ['California', 'Texas', 'Florida', 'New York', 'Illinois']
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate data
    data = []
    np.random.seed(42)
    
    for state in states:
        # Base trend and seasonality parameters
        trend_rate = np.random.uniform(0.5, 1.5)
        seasonal_amplitude = np.random.uniform(500, 1500)
        base_sales = np.random.uniform(2000, 5000)
        
        for i, date in enumerate(date_range):
            # Trend component
            trend = base_sales + (i * trend_rate)
            
            # Seasonal component (yearly pattern)
            day_of_year = date.dayofyear
            seasonal = seasonal_amplitude * np.sin(2 * np.pi * day_of_year / 365)
            
            # Weekly pattern (higher sales on weekends)
            if date.dayofweek >= 4:  # Friday, Saturday
                weekly = 200
            else:
                weekly = 0
            
            # Random noise
            noise = np.random.normal(0, 300)
            
            # Calculate sales
            sales = trend + seasonal + weekly + noise
            sales = max(sales, 100)  # Ensure positive sales
            
            # Introduce missing values (5% chance)
            if np.random.random() > 0.95:
                sales = np.nan
            
            data.append({
                'date': date,
                'state': state,
                'sales': sales
            })
    
    df = pd.DataFrame(data)
    
    # Introduce missing dates for some states (realistic data gaps)
    # Remove some dates to simulate data collection gaps
    mask = (df['date'].dt.dayofweek == 2) & (df['state'] == 'Texas') & (df['date'].dt.month == 6)
    df = df[~mask].reset_index(drop=True)
    
    return df.sort_values(['state', 'date']).reset_index(drop=True)


if __name__ == "__main__":
    # Generate and save data
    df = generate_sales_data()
    df.to_csv('sales_data.csv', index=False)
    print(f"Generated {len(df)} records across states")
    print(f"\nData shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nData info:")
    print(df.info())
    print(f"\nMissing values:")
    print(df.isnull().sum())
