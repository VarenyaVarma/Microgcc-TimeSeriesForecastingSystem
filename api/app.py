"""
FastAPI backend for the forecasting system
Provides REST API endpoints for predictions
"""
import os
import logging
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import uvicorn

from src.preprocessing import preprocess_pipeline, get_state_data
from src.feature_engineering import feature_engineering_pipeline
from src.predict import generate_all_predictions
from src.utils import setup_logger, load_model
from src.config import (RAW_DATA_FILE, MODELS_DIR, STATE_COLUMN, 
                        DATE_COLUMN, TARGET_COLUMN, FORECAST_HORIZON,
                        API_HOST, API_PORT, API_RELOAD, METRICS_DIR)

# Setup logger
logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Time Series Forecasting API",
    description="Production-grade forecasting system for multi-state sales prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Pydantic models for request/response
class ForecastPoint(BaseModel):
    """Single forecast data point"""
    date: str
    forecast: float
    forecast_lower: float
    forecast_upper: float
    confidence: str = "95%"


class ForecastResponse(BaseModel):
    """Forecast response model"""
    state: str
    best_model: str
    forecast_horizon_days: int
    forecast_horizon_weeks: int
    forecast_start_date: str
    forecast_end_date: str
    total_predictions: int
    predictions: List[ForecastPoint]
    metadata: Dict


class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    timestamp: str
    version: str
    models_available: List[str]


class ModelComparisonResponse(BaseModel):
    """Model comparison response"""
    state: str
    models: Dict
    best_model: str
    metric_used: str


# Global variables
df_processed = None
best_models_cache = {}


def load_pipeline_data():
    """Load and process data for API"""
    global df_processed
    
    logger.info("Loading pipeline data...")
    
    try:
        # Load raw data
        if not os.path.exists(RAW_DATA_FILE):
            logger.error(f"Raw data file not found: {RAW_DATA_FILE}")
            return False
        
        # Preprocess
        df = preprocess_pipeline(RAW_DATA_FILE)
        
        # Feature engineering
        df_processed = feature_engineering_pipeline(df)
        
        logger.info(f"Pipeline data loaded: {df_processed.shape}")
        return True
    
    except Exception as e:
        logger.error(f"Error loading pipeline data: {str(e)}")
        return False


def get_available_states():
    """Get list of available states"""
    if df_processed is None:
        return []
    return df_processed[STATE_COLUMN].unique().tolist()


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting API...")
    load_pipeline_data()
    logger.info("API ready")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthResponse with API status
    """
    available_models = []
    if df_processed is not None:
        for state in get_available_states()[:1]:  # Check first state
            for model_type in ['arima', 'prophet', 'xgboost', 'lstm']:
                model_file = f'{state}_{model_type}_model.pkl'
                if os.path.exists(os.path.join(MODELS_DIR, model_file)):
                    if model_type not in available_models:
                        available_models.append(model_type)
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        models_available=available_models if available_models else ["no-models-trained"]
    )


@app.get("/states")
async def list_states():
    """
    Get list of available states
    
    Returns:
        List of state names
    """
    states = get_available_states()
    
    if not states:
        raise HTTPException(
            status_code=503,
            detail="No data available. Run preprocessing pipeline first."
        )
    
    return {
        "states": sorted(states),
        "total_states": len(states)
    }


@app.get("/forecast", response_model=ForecastResponse)
async def get_forecast(state: str = Query(..., description="State name for forecast")):
    """
    Get forecast for a specific state
    
    Args:
        state: Name of the state
    
    Returns:
        ForecastResponse with predictions
    """
    # Validate input
    if df_processed is None:
        raise HTTPException(
            status_code=503,
            detail="Data not loaded. Please run preprocessing pipeline first."
        )
    
    available_states = get_available_states()
    if state not in available_states:
        raise HTTPException(
            status_code=404,
            detail=f"State '{state}' not found. Available states: {available_states}"
        )
    
    try:
        # Get state data
        state_df = get_state_data(df_processed, state)
        
        # Load trained models
        models_dict = {}
        for model_type in ['arima', 'prophet', 'xgboost', 'lstm']:
            model = load_model(model_type, state)
            if model is not None:
                models_dict[model_type] = model
        
        if not models_dict:
            raise HTTPException(
                status_code=503,
                detail=f"No trained models found for {state}. Run training pipeline first."
            )
        
        # Get best model
        best_model = best_models_cache.get(state, None)
        
        # Generate predictions
        forecast_result = generate_all_predictions(state, df_processed, models_dict, best_model)
        
        if forecast_result['final_prediction'] is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate forecast for {state}"
            )
        
        # Create response
        last_date = state_df[DATE_COLUMN].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=FORECAST_HORIZON,
            freq='D'
        )
        
        predictions_list = []
        forecast_values = forecast_result['final_prediction']
        
        for date, value in zip(forecast_dates, forecast_values):
            predictions_list.append(ForecastPoint(
                date=date.strftime('%Y-%m-%d'),
                forecast=float(value),
                forecast_lower=float(value * 0.95),
                forecast_upper=float(value * 1.05),
                confidence="95%"
            ))
        
        return ForecastResponse(
            state=state,
            best_model=best_model or "ensemble",
            forecast_horizon_days=FORECAST_HORIZON,
            forecast_horizon_weeks=FORECAST_HORIZON // 7,
            forecast_start_date=forecast_dates[0].strftime('%Y-%m-%d'),
            forecast_end_date=forecast_dates[-1].strftime('%Y-%m-%d'),
            total_predictions=len(predictions_list),
            predictions=predictions_list,
            metadata={
                "models_used": list(models_dict.keys()),
                "data_points_in_training": len(state_df),
                "last_historical_date": last_date.strftime('%Y-%m-%d'),
                "api_version": "1.0.0",
                "generated_at": datetime.now().isoformat()
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast for {state}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/model-comparison")
async def get_model_comparison(state: str = Query(..., description="State name")):
    """
    Get model comparison metrics for a state
    
    Args:
        state: State name
    
    Returns:
        Dictionary with model metrics
    """
    # Check if state is valid
    if df_processed is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    available_states = get_available_states()
    if state not in available_states:
        raise HTTPException(
            status_code=404,
            detail=f"State '{state}' not found"
        )
    
    try:
        # Load metrics if available
        metrics_file = os.path.join(METRICS_DIR, f'{state}_metrics.json')
        
        if not os.path.exists(metrics_file):
            raise HTTPException(
                status_code=404,
                detail=f"Metrics not found for {state}. Run evaluation first."
            )
        
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        return {
            "state": state,
            "metrics": metrics,
            "generated_at": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batch-forecast")
async def get_batch_forecast():
    """
    Get forecasts for all states
    
    Returns:
        Dictionary of forecasts for each state
    """
    if df_processed is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    states = get_available_states()
    results = {}
    
    try:
        for state in states:
            try:
                forecast_response = await get_forecast(state=state)
                results[state] = forecast_response.dict()
            except Exception as e:
                logger.warning(f"Error forecasting for {state}: {str(e)}")
                results[state] = {"error": str(e)}
        
        return {
            "total_states": len(states),
            "successful": sum(1 for v in results.values() if "error" not in v),
            "forecasts": results,
            "generated_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in batch forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data-info")
async def get_data_info():
    """
    Get information about loaded data
    
    Returns:
        Data statistics and info
    """
    if df_processed is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return {
        "total_records": len(df_processed),
        "total_states": df_processed[STATE_COLUMN].nunique(),
        "states": sorted(get_available_states()),
        "date_range": {
            "start": df_processed[DATE_COLUMN].min().strftime('%Y-%m-%d'),
            "end": df_processed[DATE_COLUMN].max().strftime('%Y-%m-%d')
        },
        "total_features": len(df_processed.columns),
        "features": df_processed.columns.tolist(),
        "sample_size_per_state": {
            state: int(df_processed[df_processed[STATE_COLUMN] == state].shape[0])
            for state in get_available_states()
        }
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Time Series Forecasting System",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "states": "/states",
            "forecast": "/forecast?state={state}",
            "batch_forecast": "/batch-forecast",
            "model_comparison": "/model-comparison?state={state}",
            "data_info": "/data-info"
        }
    }


def run_api(host: str = API_HOST, port: int = API_PORT, reload: bool = API_RELOAD):
    """
    Run the FastAPI server
    
    Args:
        host: Host address
        port: Port number
        reload: Enable auto-reload
    """
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_api()
