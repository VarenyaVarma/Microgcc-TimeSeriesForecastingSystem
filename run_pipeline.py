"""
Main pipeline runner for the forecasting system
Orchestrates preprocessing, feature engineering, training, evaluation, and prediction
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

from src.preprocessing import preprocess_pipeline, split_train_test, get_state_data
from src.feature_engineering import feature_engineering_pipeline, scale_features
from src.train_models import train_all_models
from src.evaluate import evaluate_model, compare_models, select_best_model, generate_evaluation_report
from src.predict import generate_all_predictions, create_forecast_dataframe
from src.utils import (setup_logger, create_timestamp, save_metrics, 
                       create_test_data_csv, load_model)
from src.config import (DATA_DIR, RAW_DATA_FILE, TARGET_COLUMN, STATE_COLUMN, 
                        DATE_COLUMN, VALIDATION_SIZE, TRAIN_TEST_SPLIT_DATE,
                        FORECAST_HORIZON, FORECASTS_DIR, PLOTS_DIR, METRICS_DIR)

logger = setup_logger(__name__)


def run_full_pipeline():
    """
    Execute the complete forecasting pipeline
    """
    logger.info("\n" + "=" * 80)
    logger.info("STARTING FORECASTING PIPELINE")
    logger.info("=" * 80)
    
    pipeline_results = {}
    
    # Step 1: Generate/Load Data
    logger.info("\n[STEP 1] DATA LOADING")
    logger.info("-" * 80)
    
    if not os.path.exists(RAW_DATA_FILE):
        logger.info("Creating test data...")
        df_raw = create_test_data_csv()
    else:
        logger.info("Loading existing data...")
        df_raw = pd.read_csv(RAW_DATA_FILE)
        df_raw[DATE_COLUMN] = pd.to_datetime(df_raw[DATE_COLUMN])
    
    logger.info(f"Data shape: {df_raw.shape}")
    pipeline_results['data_shape'] = df_raw.shape
    
    # Step 2: Preprocessing
    logger.info("\n[STEP 2] PREPROCESSING")
    logger.info("-" * 80)
    
    df_preprocessed = preprocess_pipeline(RAW_DATA_FILE)
    pipeline_results['preprocessed_shape'] = df_preprocessed.shape
    
    # Step 3: Feature Engineering
    logger.info("\n[STEP 3] FEATURE ENGINEERING")
    logger.info("-" * 80)
    
    df_features = feature_engineering_pipeline(df_preprocessed)
    pipeline_results['features_shape'] = df_features.shape
    
    # Step 4: Train-Test Split (Temporal Split)
    logger.info("\n[STEP 4] TRAIN-TEST SPLIT")
    logger.info("-" * 80)
    
    train_df_full, test_df_full = split_train_test(df_features, split_date=TRAIN_TEST_SPLIT_DATE)
    
    # Step 5: Model Training and Validation
    logger.info("\n[STEP 5] MODEL TRAINING AND VALIDATION")
    logger.info("-" * 80)
    
    all_results = {}
    best_models = {}
    trained_models = {}
    
    for state in df_features[STATE_COLUMN].unique():
        logger.info(f"\n{'*'*60}")
        logger.info(f"Processing State: {state}")
        logger.info(f"{'*'*60}")
        
        # Get state-wise data
        train_state = get_state_data(train_df_full, state)
        test_state = get_state_data(test_df_full, state)
        
        logger.info(f"Train samples: {len(train_state)}, Test samples: {len(test_state)}")
        
        if len(train_state) < 100:
            logger.warning(f"Not enough training data for {state}. Skipping.")
            continue
        
        # Create validation set from train set (temporal split within train)
        val_split_idx = len(train_state) - VALIDATION_SIZE
        train_val = train_state.iloc[:val_split_idx]
        val_val = train_state.iloc[val_split_idx:]
        
        # Train all models
        models = train_all_models(train_val, val_val, state)
        trained_models[state] = models
        
        # Evaluations on test set
        state_results = {}
        
        for model_name, model in models.items():
            if model is None:
                logger.warning(f"Model {model_name} is None for {state}")
                continue
            
            try:
                # Get predictions on test set
                if model_name == 'arima':
                    from src.train_models import forecast_arima
                    y_pred = forecast_arima(model, steps=len(test_state))
                    if y_pred is not None and len(y_pred) > len(test_state):
                        y_pred = y_pred[:len(test_state)]
                
                elif model_name == 'prophet':
                    from src.train_models import forecast_prophet
                    y_pred = forecast_prophet(model, steps=len(test_state))
                    if y_pred is not None and len(y_pred) > len(test_state):
                        y_pred = y_pred[:len(test_state)]
                
                elif model_name == 'xgboost':
                    from src.train_models import forecast_xgboost
                    y_pred = forecast_xgboost(model, test_state, steps=len(test_state))
                    if y_pred is not None and len(y_pred) > len(test_state):
                        y_pred = y_pred[:len(test_state)]
                
                elif model_name == 'lstm':
                    from src.train_models import forecast_lstm
                    from sklearn.preprocessing import StandardScaler
                    
                    sales_data = train_state[TARGET_COLUMN].values
                    scaler = StandardScaler()
                    sales_scaled = scaler.fit_transform(sales_data.reshape(-1, 1))
                    
                    y_pred = forecast_lstm(model, sales_scaled, steps=len(test_state))
                    if y_pred is not None and len(y_pred) > len(test_state):
                        y_pred = y_pred[:len(test_state)]
                
                if y_pred is not None and len(y_pred) == len(test_state):
                    # Evaluate
                    y_true = test_state[TARGET_COLUMN].values
                    metrics = evaluate_model(y_true, y_pred, model_name, state)
                    state_results[model_name] = metrics
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} for {state}: {str(e)}")
                continue
        
        # Select best model
        if state_results:
            comparison_df = compare_models(state_results)
            best_model = select_best_model(comparison_df, metric='rmse')
            best_models[state] = best_model
            all_results[state] = state_results
    
    pipeline_results['best_models'] = best_models
    
    # Step 6: Generate Evaluation Report
    logger.info("\n[STEP 6] EVALUATION REPORT")
    logger.info("-" * 80)
    
    if all_results:
        evaluation_report = generate_evaluation_report(all_results)
        pipeline_results['evaluation'] = evaluation_report
        
        # Save evaluation metrics
        for state, evaluation in evaluation_report.items():
            save_metrics(evaluation, state)
    
    # Step 7: Forecasting
    logger.info("\n[STEP 7] FORECASTING (8 weeks ahead)")
    logger.info("-" * 80)
    
    all_forecasts = {}
    
    for state in df_features[STATE_COLUMN].unique():
        logger.info(f"\nForecasting for {state}...")
        
        best_model = best_models.get(state)
        models = trained_models.get(state, {})
        
        # Generate predictions
        forecast_result = generate_all_predictions(state, df_features, models, best_model)
        
        if forecast_result['final_prediction'] is not None:
            # Create forecast dataframe
            state_df = get_state_data(df_features, state)
            last_date = state_df[DATE_COLUMN].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                        periods=FORECAST_HORIZON, freq='D')
            
            forecast_df = pd.DataFrame({
                DATE_COLUMN: future_dates,
                STATE_COLUMN: state,
                'forecast': forecast_result['final_prediction'],
                'forecast_lower': forecast_result['final_prediction'] * 0.95,
                'forecast_upper': forecast_result['final_prediction'] * 1.05,
                'model_used': best_model or 'ensemble'
            })
            
            all_forecasts[state] = forecast_df
            
            # Save forecast
            forecast_file = os.path.join(FORECASTS_DIR, f'{state}_forecast.csv')
            forecast_df.to_csv(forecast_file, index=False)
            logger.info(f"Forecast saved to {forecast_file}")
    
    pipeline_results['forecasts'] = all_forecasts
    
    # Step 8: Visualization
    logger.info("\n[STEP 8] VISUALIZATION")
    logger.info("-" * 80)
    
    try:
        create_visualizations(df_features, all_forecasts, all_results)
    except Exception as e:
        logger.warning(f"Error creating visualizations: {str(e)}")
    
    # Step 9: Summary Report
    logger.info("\n[STEP 9] PIPELINE SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"Total states processed: {len(best_models)}")
    logger.info(f"Best models selected: {len(best_models)}")
    logger.info(f"Forecasts generated: {len(all_forecasts)}")
    logger.info(f"Forecast horizon: {FORECAST_HORIZON} days (8 weeks)")
    
    logger.info(f"\nBest model selection:")
    for state, model in best_models.items():
        logger.info(f"  {state}: {model.upper()}")
    
    # Save pipeline results
    results_file = os.path.join(METRICS_DIR, f'pipeline_results_{create_timestamp()}.json')
    
    # Convert non-serializable objects
    results_to_save = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': str(pipeline_results.get('data_shape')),
        'preprocessed_shape': str(pipeline_results.get('preprocessed_shape')),
        'features_shape': str(pipeline_results.get('features_shape')),
        'best_models': best_models,
        'states': list(df_features[STATE_COLUMN].unique()),
        'forecast_horizon_days': FORECAST_HORIZON,
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    
    logger.info(f"\nPipeline results saved to: {results_file}")
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    
    return pipeline_results


def create_visualizations(df_full, forecasts, evaluation_results):
    """
    Create visualizations for results
    
    Args:
        df_full: Full processed data
        forecasts: Dictionary of forecasts per state
        evaluation_results: Dictionary of evaluation results
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    logger.info("Creating visualizations...")
    
    # 1. Sales trends and forecasts
    for state, forecast_df in forecasts.items():
        try:
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Get historical data
            state_df = df_full[df_full[STATE_COLUMN] == state].copy()
            
            # Plot historical sales
            ax.plot(state_df[DATE_COLUMN], state_df[TARGET_COLUMN], 
                   'b-', label='Historical Sales', linewidth=2)
            
            # Plot forecast
            ax.plot(forecast_df[DATE_COLUMN], forecast_df['forecast'], 
                   'r--', label='Forecast', linewidth=2)
            
            # Plot confidence interval
            ax.fill_between(forecast_df[DATE_COLUMN],
                           forecast_df['forecast_lower'],
                           forecast_df['forecast_upper'],
                           alpha=0.2, color='red', label='95% Confidence Interval')
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Sales', fontsize=12)
            ax.set_title(f'Sales Forecast for {state}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_file = os.path.join(PLOTS_DIR, f'{state}_forecast.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved forecast plot: {plot_file}")
            plt.close()
        
        except Exception as e:
            logger.warning(f"Error creating forecast plot for {state}: {str(e)}")
    
    # 2. Model comparison plot
    if evaluation_results:
        try:
            from src.evaluate import plot_model_comparison
            
            plot_file = os.path.join(PLOTS_DIR, 'model_comparison.png')
            plot_model_comparison(evaluation_results, plot_file)
        except Exception as e:
            logger.warning(f"Error creating model comparison plot: {str(e)}")
    
    logger.info("Visualization complete")


if __name__ == '__main__':
    try:
        results = run_full_pipeline()
        logger.info("\nPipeline execution successful!")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
