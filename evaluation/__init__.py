"""
Model evaluation and visualization utilities.

This module provides comprehensive tools for evaluating machine learning models
with built-in visualization capabilities for both classification and regression tasks.
"""

from .model_evaluation import (
    evaluate_classification_model,
    evaluate_regression_model,
    compare_models,
)

__all__ = [
    'evaluate_classification_model',
    'evaluate_regression_model',
    'compare_models',
]

