"""
Gradient Boosting Decision Tree (GBDT) example.

Gradient boosting is an ensemble machine learning technique that constructs a
strong learner by sequentially adding weak learners (e.g. decision trees).  In
classification tasks the model attempts to minimise a differentiable loss
function by fitting each new tree on the negative gradient of the loss with
respect to the current ensemble's predictions.

This script uses scikit‑learn's GradientBoostingClassifier to train a model on
a synthetic binary classification problem and prints the accuracy on a hold‑out
test set.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_gbdt(n_samples: int = 1000, seed: int = 0):
    """
    Generate synthetic data and train a gradient boosting classifier.

    Returns
    -------
    model : GradientBoostingClassifier
        Trained classifier.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    """
    X, y = make_classification(n_samples=n_samples,
                               n_features=10,
                               n_informative=5,
                               n_redundant=2,
                               random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    return model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train_gbdt()
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}")
