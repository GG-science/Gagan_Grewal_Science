"""
ML Model Evaluation Module with Visualizations

This module provides comprehensive evaluation tools for machine learning models
with built-in visualization capabilities. It supports both classification and
regression tasks.

USAGE:
    # For classification models:
    from evaluation.model_evaluation import evaluate_classification_model
    
    model = train_your_classifier()
    X_test, y_test = get_test_data()
    evaluate_classification_model(model, X_test, y_test, model_name="My Model")
    
    # For regression models:
    from evaluation.model_evaluation import evaluate_regression_model
    
    model = train_your_regressor()
    X_test, y_test = get_test_data()
    evaluate_regression_model(model, X_test, y_test, model_name="My Model")
"""

import sys
from pathlib import Path

# Add parent directory to path to import data module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from typing import Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def evaluate_classification_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    save_plots: bool = True,
    output_dir: str = "evaluation/plots"
) -> dict:
    """
    Comprehensive evaluation for classification models with visualizations.
    
    This function generates:
    - Confusion matrix heatmap
    - ROC curve (for binary classification)
    - Classification metrics summary
    - Feature importance plot (if available)
    
    Parameters
    ----------
    model : sklearn classifier
        Trained classification model with predict() and predict_proba() methods
    X_test : np.ndarray
        Test feature matrix
    y_test : np.ndarray
        True test labels
    model_name : str, optional
        Name of the model for plot titles (default: "Model")
    save_plots : bool, optional
        Whether to save plots to disk (default: True)
    output_dir : str, optional
        Directory to save plots (default: "evaluation/plots")
    
    Returns
    -------
    dict
        Dictionary containing all evaluation metrics
    
    Example
    -------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, random_state=42)
    >>> model = RandomForestClassifier().fit(X, y)
    >>> metrics = evaluate_classification_model(model, X, y, "RF")
    """
    # Create output directory if saving plots
    if save_plots:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Check if model supports probability prediction (for ROC curve)
    has_proba = hasattr(model, 'predict_proba')
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Determine if binary or multiclass
    n_classes = len(np.unique(y_test))
    is_binary = n_classes == 2
    
    # Create figure with subplots
    if is_binary and has_proba:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test),
        ax=axes[0]
    )
    axes[0].set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    
    # 2. ROC Curve (only for binary classification with probabilities)
    if is_binary and has_proba:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate', fontsize=12)
        axes[1].set_ylabel('True Positive Rate', fontsize=12)
        axes[1].set_title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold')
        axes[1].legend(loc="lower right", fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Metrics summary
        metrics_text = f"""
        Accuracy: {accuracy:.3f}
        Precision: {precision:.3f}
        Recall: {recall:.3f}
        F1-Score: {f1:.3f}
        ROC-AUC: {roc_auc:.3f}
        """
        axes[2].text(0.1, 0.5, metrics_text, fontsize=12, 
                    verticalalignment='center', family='monospace')
        axes[2].set_title(f'{model_name} - Metrics Summary', fontsize=14, fontweight='bold')
        axes[2].axis('off')
    else:
        # Metrics summary for multiclass or models without probabilities
        metrics_text = f"""
        Accuracy: {accuracy:.3f}
        Precision: {precision:.3f}
        Recall: {recall:.3f}
        F1-Score: {f1:.3f}
        """
        axes[1].text(0.1, 0.5, metrics_text, fontsize=12, 
                    verticalalignment='center', family='monospace')
        axes[1].set_title(f'{model_name} - Metrics Summary', fontsize=14, fontweight='bold')
        axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    if save_plots:
        plot_path = f"{output_dir}/{model_name.lower().replace(' ', '_')}_classification.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {plot_path}")
    
    plt.show()
    
    # Print classification report
    print("\n" + "="*60)
    print(f"Classification Report for {model_name}")
    print("="*60)
    print(classification_report(y_test, y_pred))
    
    # Return metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }
    
    if is_binary and has_proba:
        metrics['roc_auc'] = roc_auc
    
    return metrics


def evaluate_regression_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    save_plots: bool = True,
    output_dir: str = "evaluation/plots"
) -> dict:
    """
    Comprehensive evaluation for regression models with visualizations.
    
    This function generates:
    - Predicted vs Actual scatter plot
    - Residual plot
    - Metrics summary
    
    Parameters
    ----------
    model : sklearn regressor
        Trained regression model with predict() method
    X_test : np.ndarray
        Test feature matrix
    y_test : np.ndarray
        True test target values
    model_name : str, optional
        Name of the model for plot titles (default: "Model")
    save_plots : bool, optional
        Whether to save plots to disk (default: True)
    output_dir : str, optional
        Directory to save plots (default: "evaluation/plots")
    
    Returns
    -------
    dict
        Dictionary containing all evaluation metrics
    
    Example
    -------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, random_state=42)
    >>> model = RandomForestRegressor().fit(X, y)
    >>> metrics = evaluate_regression_model(model, X, y, "RF")
    """
    # Create output directory if saving plots
    if save_plots:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Predicted vs Actual
    axes[0].scatter(y_test, y_pred, alpha=0.5, s=50)
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 
                'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Values', fontsize=12)
    axes[0].set_ylabel('Predicted Values', fontsize=12)
    axes[0].set_title(f'{model_name} - Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Residual Plot
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, s=50)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title(f'{model_name} - Residual Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Metrics Summary
    metrics_text = f"""
    R² Score: {r2:.3f}
    RMSE: {rmse:.3f}
    MAE: {mae:.3f}
    MSE: {mse:.3f}
    """
    axes[2].text(0.1, 0.5, metrics_text, fontsize=12, 
                verticalalignment='center', family='monospace')
    axes[2].set_title(f'{model_name} - Metrics Summary', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    if save_plots:
        plot_path = f"{output_dir}/{model_name.lower().replace(' ', '_')}_regression.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {plot_path}")
    
    plt.show()
    
    # Print metrics
    print("\n" + "="*60)
    print(f"Regression Metrics for {model_name}")
    print("="*60)
    print(f"R² Score:  {r2:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print(f"MAE:       {mae:.4f}")
    print(f"MSE:       {mse:.4f}")
    print("="*60)
    
    # Return metrics dictionary
    return {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'mse': mse
    }


def compare_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = 'classification',
    save_plots: bool = True,
    output_dir: str = "evaluation/plots"
) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.
    
    This function is useful when you want to evaluate and compare
    multiple models on the same test set. It generates comparison
    plots and a summary table.
    
    Parameters
    ----------
    models : dict
        Dictionary mapping model names to model objects
        Example: {'Random Forest': rf_model, 'SVM': svm_model}
    X_test : np.ndarray
        Test feature matrix
    y_test : np.ndarray
        True test labels/values
    task_type : str, optional
        'classification' or 'regression' (default: 'classification')
    save_plots : bool, optional
        Whether to save plots to disk (default: True)
    output_dir : str, optional
        Directory to save plots (default: "evaluation/plots")
    
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for all models
    
    Example
    -------
    >>> models = {
    ...     'RF': RandomForestClassifier(),
    ...     'SVM': SVC(),
    ...     'LR': LogisticRegression()
    ... }
    >>> # Train models...
    >>> comparison = compare_models(models, X_test, y_test)
    """
    if save_plots:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for name, model in models.items():
        if task_type == 'classification':
            metrics = evaluate_classification_model(
                model, X_test, y_test, name, save_plots=False
            )
        else:
            metrics = evaluate_regression_model(
                model, X_test, y_test, name, save_plots=False
            )
        metrics['model'] = name
        results.append(metrics)
    
    # Create comparison DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('model')
    
    # Create comparison plot
    if task_type == 'classification':
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    else:
        metrics_to_plot = ['r2_score', 'rmse', 'mae']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5*len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics_to_plot):
        if metric in df_results.columns:
            df_results[metric].plot(kind='bar', ax=axes[idx], color='steelblue')
            axes[idx].set_title(f'{metric.replace("_", " ").title()}', 
                              fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_plots:
        plot_path = f"{output_dir}/model_comparison_{task_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {plot_path}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    print(df_results.round(4))
    print("="*60)
    
    return df_results


if __name__ == "__main__":
    """
    Example usage of the evaluation module.
    
    This demonstrates how to use the evaluation functions with
    synthetic data. Run this script to see example visualizations.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from data import generate_classification_data, generate_continuous_causal_data
    from sklearn.model_selection import train_test_split
    
    print("="*60)
    print("ML Model Evaluation Module - Example Usage")
    print("="*60)
    
    # Example 1: Classification Model Evaluation
    print("\n1. Evaluating Classification Model...")
    X, y = generate_classification_data(n_samples=1000, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_model.fit(X_train, y_train)
    
    clf_metrics = evaluate_classification_model(
        clf_model, X_test, y_test, 
        model_name="Random Forest Classifier",
        save_plots=True
    )
    
    # Example 2: Regression Model Evaluation
    print("\n2. Evaluating Regression Model...")
    df, _ = generate_continuous_causal_data(n=600, seed=21)
    X_reg = df[['x1', 'x2']].values
    y_reg = df['y'].values
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train_reg, y_train_reg)
    
    reg_metrics = evaluate_regression_model(
        reg_model, X_test_reg, y_test_reg,
        model_name="Random Forest Regressor",
        save_plots=True
    )
    
    # Example 3: Model Comparison
    print("\n3. Comparing Multiple Models...")
    X_train_comp, X_test_comp, y_train_comp, y_test_comp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    
    rf.fit(X_train_comp, y_train_comp)
    lr.fit(X_train_comp, y_train_comp)
    
    models_to_compare = {
        'Random Forest': rf,
        'Logistic Regression': lr
    }
    
    comparison_df = compare_models(
        models_to_compare, X_test_comp, y_test_comp,
        task_type='classification',
        save_plots=True
    )
    
    print("\n✓ All evaluations complete!")

