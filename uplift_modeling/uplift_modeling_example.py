"""
Simple uplift modeling example using a two‑model approach.

Uplift modelling (also known as incremental response modelling) seeks to
estimate the difference in probability of an outcome under treatment versus
control for individual units.  One straightforward approach is to train two
separate models: one on the treated group and one on the control group.  The
uplift for a given observation is then the difference between the predicted
probabilities of the outcome from the two models.

This example generates a synthetic binary classification problem with a binary
treatment indicator and trains logistic regression models separately on the
treated and control subsets.  The predicted uplift is computed for each sample.
"""

import sys
from pathlib import Path

# Add parent directory to path to import data module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data import generate_uplift_data

def train_two_models(df: pd.DataFrame):
    """
    Train separate logistic regression models on treated and control groups.

    Returns
    -------
    model_t : LogisticRegression
        Model trained on treated samples.
    model_c : LogisticRegression
        Model trained on control samples.
    """
    model_t = LogisticRegression(max_iter=1000)
    model_c = LogisticRegression(max_iter=1000)
    # treated data
    Xt = df[df['treatment'] == 1][['x1','x2']]
    yt = df[df['treatment'] == 1]['y']
    model_t.fit(Xt, yt)
    # control data
    Xc = df[df['treatment'] == 0][['x1','x2']]
    yc = df[df['treatment'] == 0]['y']
    model_c.fit(Xc, yc)
    return model_t, model_c

def compute_uplift(model_t, model_c, X: pd.DataFrame):
    """
    Compute uplift as the difference in predicted probabilities from the two models.

    Parameters
    ----------
    model_t : classifier with predict_proba
    model_c : classifier with predict_proba
    X : pd.DataFrame

    Returns
    -------
    uplift : np.ndarray
    """
    p_t = model_t.predict_proba(X)[:, 1]
    p_c = model_c.predict_proba(X)[:, 1]
    return p_t - p_c

if __name__ == "__main__":
    df = generate_uplift_data()
    model_t, model_c = train_two_models(df)
    uplift = compute_uplift(model_t, model_c, df[['x1','x2']])
    print("Uplift scores (first 10):", uplift[:10])
