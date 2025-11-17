"""
Propensity Score Matching (PSM) example.

Propensity score matching is a technique to reduce selection bias when
estimating causal effects using observational data.  The idea is to model the
probability of receiving treatment (the propensity score) as a function of
observed covariates and then match treated units to control units with similar
propensities.

This script constructs a synthetic dataset with a binary treatment and a
continuous outcome influenced by both the treatment and covariates.  It fits a
logistic regression to estimate propensity scores and then performs nearest
neighbour matching on the scores.  Finally it computes an average treatment
effect on the matched sample.
"""

import sys
from pathlib import Path

# Add parent directory to path to import data module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from data import generate_psm_data

def estimate_propensity(data: pd.DataFrame):
    """
    Fit a logistic regression model to estimate propensity scores.
    """
    model = LogisticRegression(max_iter=1000)
    X = data[['x1', 'x2']]
    y = data['treatment']
    model.fit(X, y)
    prop_scores = model.predict_proba(X)[:, 1]
    return prop_scores

def match_and_estimate_ate(data: pd.DataFrame, prop_scores: np.ndarray):
    """
    Perform nearest neighbour matching on the propensity scores and estimate ATE.

    Returns
    -------
    ate : float
        Estimated average treatment effect.
    """
    # separate treated and control
    treated = data[data['treatment'] == 1].copy()
    control = data[data['treatment'] == 0].copy()
    treated_scores = prop_scores[data['treatment'] == 1].reshape(-1, 1)
    control_scores = prop_scores[data['treatment'] == 0].reshape(-1, 1)
    # nearest neighbour matching
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control_scores)
    distances, indices = nn.kneighbors(treated_scores)
    matched_controls = control.iloc[indices.flatten()]
    # compute treatment effect for matched pairs
    effect = treated['outcome'].values - matched_controls['outcome'].values
    ate = np.mean(effect)
    return ate

if __name__ == "__main__":
    data = generate_psm_data()
    prop_scores = estimate_propensity(data)
    ate = match_and_estimate_ate(data, prop_scores)
    print(f"Estimated ATE (PSM): {ate:.3f}")
