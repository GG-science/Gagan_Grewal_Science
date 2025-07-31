# Gagan Grewal Science

This repository contains a collection of small Python scripts demonstrating
fundamental concepts in machine learning and causal inference.  Each script
illustrates a technique on a self‑contained toy dataset so that you can
understand how the algorithms work without needing a large or proprietary data
source.

The examples included are:

| script | description |
| --- | --- |
| `faiss_example.py` | builds a simple FAISS index from random vectors and performs nearest‑neighbour search. FAISS is a library for fast approximate similarity search on large vector collections. |
| `psm_example.py` | demonstrates propensity score matching (PSM) for estimating the effect of a binary treatment using logistic regression and nearest neighbour matching on synthetic data. |
| `uplift_modeling_example.py` | implements a two‑model approach to uplift modelling to estimate heterogeneous treatment effects. Two separate models (treatment and control) are trained on a synthetic dataset and the difference in predicted probabilities is used as the uplift. |
| `s_learner_example.py` | implements the S‑learner from causal inference using a single gradient boosting regressor that takes treatment as an input feature.  Conditional average treatment effects (CATE) are estimated by toggling the treatment variable and computing the difference in predictions. |
| `gbdt_example.py` | trains a gradient boosting decision tree classifier on a toy classification problem and evaluates its accuracy. |

## Requirements

The examples rely only on standard Python scientific packages such as
`numpy`, `pandas` and `scikit‑learn`.  The `faiss_example.py` script also
requires the [`faiss`](https://github.com/facebookresearch/faiss) library to be
installed. If it is not available in your environment the example will skip
execution and print a warning instead of raising an error.

Each script can be executed directly:

```bash
python faiss_example.py
python psm_example.py
python uplift_modeling_example.py
python s_learner_example.py
python gbdt_example.py
```

Since these scripts create random synthetic data on every run your numerical
results may differ from those shown in the comments, but the overall
interpretation should remain the same.
