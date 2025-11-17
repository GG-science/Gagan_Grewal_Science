"""
S‑learner example using gradient boosting regression.

The S‑learner is a simple meta‑learner for estimating conditional average
treatment effects (CATE).  A single predictive model is trained on the entire
dataset with the treatment indicator included as a feature.  For a given unit
the CATE is estimated by computing the difference between the predicted
outcomes when the treatment indicator is set to 1 versus 0.

This script constructs a synthetic continuous outcome dataset influenced by
covariates and a treatment effect.  It trains a gradient boosting regressor on
the full data and then estimates the CATE for each observation.
"""

import sys
from pathlib import Path

# Add parent directory to path to import data module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from data import generate_continuous_causal_data

def train_s_learner(df: pd.DataFrame):
    """
    Train a gradient boosting regressor on covariates and treatment indicator.
    """
    X = df[['x1','x2','treatment']]
    y = df['y']
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

def estimate_cate(model, df: pd.DataFrame):
    """
    Estimate CATE by toggling the treatment feature and computing predictions.
    """
    X0 = df[['x1','x2']].copy()
    X1 = X0.copy()
    X0['treatment'] = 0
    X1['treatment'] = 1
    # predictions under no treatment and treatment
    y0 = model.predict(X0)
    y1 = model.predict(X1)
    return y1 - y0

if __name__ == "__main__":
    df, true_effect = generate_continuous_causal_data()
    model = train_s_learner(df)
    cate = estimate_cate(model, df)
    print("Estimated CATE (first 10):", cate[:10])
    print(f"True effect used in simulation: {true_effect}")
