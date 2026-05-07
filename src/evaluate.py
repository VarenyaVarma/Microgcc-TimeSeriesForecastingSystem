"""
Evaluation module for model performance assessment
Calculates MAE, RMSE, MAPE and generates comparison reports
"""
import pandas as pd
import numpy as np
from src.utils import calculate_mae, calculate_rmse, calculate_mape, setup_logger
from src.config import TARGET_COLUMN, STATE_COLUMN

logger = setup_logger(__name__)


def evaluate_model(y_true, y_pred, model_name='', state=''):
    """
    Evaluate model predictions against true values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        state: State name
    
    Returns:
        Dictionary with evaluation metrics
    """
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    metrics = {
        'model': model_name,
        'state': state,
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'samples': len(y_true)
    }
    
    logger.info(f"{state} - {model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
    
    return metrics


def compare_models(results_dict):
    """
    Compare metrics across multiple models for a state
    
    Args:
        results_dict: Dictionary of model results {model_name: metrics}
    
    Returns:
        Sorted DataFrame with model comparison
    """
    comparison_df = pd.DataFrame(results_dict).T
    comparison_df = comparison_df.sort_values('rmse', ascending=True)
    
    logger.info("\nModel Comparison (sorted by RMSE):")
    logger.info(comparison_df.to_string())
    
    return comparison_df


def select_best_model(comparison_df, metric='rmse'):
    """
    Select best model based on specified metric
    
    Args:
        comparison_df: DataFrame with model metrics
        metric: Metric to use for selection ('rmse', 'mae', or 'mape')
    
    Returns:
        Name of best model
    """
    best_model = comparison_df[metric].idxmin()
    best_score = comparison_df.loc[best_model, metric]
    
    logger.info(f"\nBest Model: {best_model} ({metric}={best_score:.2f})")
    
    return best_model


def generate_evaluation_report(all_results):
    """
    Generate comprehensive evaluation report
    
    Args:
        all_results: Dictionary {state: {model_name: metrics}}
    
    Returns:
        Dictionary with summary statistics
    """
    logger.info("\n" + "=" * 60)
    logger.info("MODEL EVALUATION REPORT")
    logger.info("=" * 60)
    
    report = {}
    
    for state, state_results in all_results.items():
        logger.info(f"\nState: {state}")
        logger.info("-" * 40)
        
        # Create comparison dataframe
        comparison_df = compare_models(state_results)
        
        # Select best model
        best_model = select_best_model(comparison_df, metric='rmse')
        
        # Calculate ensemble metrics (average)
        ensemble_mae = np.mean([m['mae'] for m in state_results.values()])
        ensemble_rmse = np.mean([m['rmse'] for m in state_results.values()])
        ensemble_mape = np.mean([m['mape'] for m in state_results.values()])
        
        report[state] = {
            'best_model': best_model,
            'best_rmse': float(comparison_df.loc[best_model, 'rmse']),
            'best_mae': float(comparison_df.loc[best_model, 'mae']),
            'best_mape': float(comparison_df.loc[best_model, 'mape']),
            'ensemble_rmse': float(ensemble_rmse),
            'ensemble_mae': float(ensemble_mae),
            'ensemble_mape': float(ensemble_mape),
            'all_models': state_results,
            'comparison': comparison_df.to_dict('index')
        }
        
        logger.info(f"Best model: {best_model}")
        logger.info(f"Ensemble RMSE: {ensemble_rmse:.2f}")
    
    logger.info("\n" + "=" * 60)
    return report


def calculate_prediction_intervals(y_true, y_pred, confidence=95):
    """
    Calculate prediction intervals based on residuals
    
    Args:
        y_true: True values
        y_pred: Predicted values
        confidence: Confidence level (95, 99)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    residuals = y_true - y_pred
    std_error = np.std(residuals)
    
    if confidence == 95:
        z_score = 1.96
    elif confidence == 99:
        z_score = 2.576
    else:
        z_score = 1.96
    
    margin = z_score * std_error
    
    lower_bound = y_pred - margin
    upper_bound = y_pred + margin
    
    return lower_bound, upper_bound


def plot_model_comparison(all_results, output_path):
    """
    Create visualization comparing models across states
    
    Args:
        all_results: Dictionary {state: {model_name: metrics}}
        output_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    states = list(all_results.keys())
    models = set()
    for state_results in all_results.values():
        models.update(state_results.keys())
    models = sorted(list(models))
    
    # Prepare data
    rmse_data = {}
    mae_data = {}
    
    for model in models:
        rmse_vals = []
        mae_vals = []
        for state in states:
            if model in all_results[state]:
                rmse_vals.append(all_results[state][model]['rmse'])
                mae_vals.append(all_results[state][model]['mae'])
        rmse_data[model] = rmse_vals
        mae_data[model] = mae_vals
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # RMSE comparison
    x = np.arange(len(states))
    width = 0.2
    for i, model in enumerate(models):
        axes[0].bar(x + i * width, rmse_data[model], width, label=model)
    
    axes[0].set_xlabel('State')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE Comparison Across States')
    axes[0].set_xticks(x + width * (len(models) - 1) / 2)
    axes[0].set_xticklabels(states, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE comparison
    for i, model in enumerate(models):
        axes[1].bar(x + i * width, mae_data[model], width, label=model)
    
    axes[1].set_xlabel('State')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE Comparison Across States')
    axes[1].set_xticks(x + width * (len(models) - 1) / 2)
    axes[1].set_xticklabels(states, rotation=45)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison plot: {output_path}")
    plt.close()
