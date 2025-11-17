# Gagan Grewal Science

A well-structured repository containing Python scripts demonstrating fundamental concepts in machine learning, causal inference, and similarity search. Each script illustrates a technique on self-contained synthetic datasets, making it easy to understand how the algorithms work without needing large or proprietary data sources.

## Repository Structure

```
Gagan_Grewal_Science/
├── data/                    # Data ingestion and processing utilities
│   ├── __init__.py
│   └── ingestion.py         # Data loading and generation functions
├── matching/                # Matching techniques
│   ├── psm_example.py       # Propensity Score Matching
│   └── faiss_example.py     # FAISS similarity search
├── uplift_modeling/         # Uplift modeling techniques
│   ├── uplift_modeling_example.py  # Two-model uplift approach
│   └── s_learner_example.py        # S-learner for CATE estimation
├── evaluation/              # Model evaluation and visualization
│   ├── __init__.py
│   └── model_evaluation.py  # Comprehensive ML model evaluation with charts
├── utils/                   # General utilities
│   └── gbdt_example.py      # Gradient Boosting Decision Tree
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup configuration
└── README.md                # This file
```

## Features

### Data Ingestion (`data/ingestion.py`)

The data ingestion module provides utilities for:
- Loading data from various sources:
  - Local files (CSV, Excel, Parquet, JSON)
  - **S3 buckets** (requires boto3)
- Generating synthetic datasets for different use cases:
  - Propensity Score Matching data
  - Uplift modeling data
  - Continuous causal inference data
  - Classification data
  - Vector data for similarity search
- Data validation and splitting utilities

### Matching Techniques (`matching/`)

#### Propensity Score Matching (`psm_example.py`)
Demonstrates propensity score matching (PSM) for estimating the effect of a binary treatment using logistic regression and nearest neighbour matching on synthetic data. PSM reduces selection bias when estimating causal effects using observational data.

**Usage:**
```bash
python matching/psm_example.py
```

#### FAISS Similarity Search (`faiss_example.py`)
Builds a simple FAISS index from random vectors and performs nearest-neighbour search. FAISS is optimized for large datasets and high-dimensional spaces where brute force search becomes too slow.

**Usage:**
```bash
python matching/faiss_example.py
```

### Uplift Modeling (`uplift_modeling/`)

#### Two-Model Uplift Approach (`uplift_modeling_example.py`)
Implements a two-model approach to uplift modeling to estimate heterogeneous treatment effects. Two separate models (treatment and control) are trained on a synthetic dataset, and the difference in predicted probabilities is used as the uplift.

**Usage:**
```bash
python uplift_modeling/uplift_modeling_example.py
```

#### S-Learner (`s_learner_example.py`)
Implements the S-learner from causal inference using a single gradient boosting regressor that takes treatment as an input feature. Conditional average treatment effects (CATE) are estimated by toggling the treatment variable and computing the difference in predictions.

**Usage:**
```bash
python uplift_modeling/s_learner_example.py
```

### General Utilities (`utils/`)

#### Gradient Boosting Decision Tree (`gbdt_example.py`)
Trains a gradient boosting decision tree classifier on a toy classification problem and evaluates its accuracy. Gradient boosting is an ensemble technique that constructs a strong learner by sequentially adding weak learners.

**Usage:**
```bash
python utils/gbdt_example.py
```

### Model Evaluation (`evaluation/`)

#### Model Evaluation with Visualizations (`model_evaluation.py`)
Comprehensive evaluation tools for machine learning models with built-in visualization capabilities. Supports both classification and regression tasks with automatic generation of diagnostic charts.

**Features:**
- **Classification Evaluation:**
  - Confusion matrix heatmap
  - ROC curve (for binary classification)
  - Classification metrics summary (accuracy, precision, recall, F1-score)
  
- **Regression Evaluation:**
  - Predicted vs Actual scatter plot
  - Residual plot
  - Regression metrics summary (R², RMSE, MAE, MSE)
  
- **Model Comparison:**
  - Side-by-side comparison of multiple models
  - Summary table with all metrics
  - Comparison visualizations

**Usage:**
```python
from evaluation import evaluate_classification_model, evaluate_regression_model, compare_models

# Evaluate a classification model
metrics = evaluate_classification_model(
    model, X_test, y_test, 
    model_name="My Classifier",
    save_plots=True
)

# Evaluate a regression model
metrics = evaluate_regression_model(
    model, X_test, y_test,
    model_name="My Regressor",
    save_plots=True
)

# Compare multiple models
models = {'RF': rf_model, 'SVM': svm_model, 'LR': lr_model}
comparison_df = compare_models(models, X_test, y_test, task_type='classification')
```

**Run example:**
```bash
python evaluation/model_evaluation.py
```

**Key Usability Features:**
- Automatic plot generation and saving
- Clear, informative visualizations with proper labels
- Comprehensive metrics reporting
- Easy integration with any scikit-learn compatible model
- Detailed docstrings and comments for easy understanding
- Flexible output options (display, save, or both)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Gagan_Grewal_Science
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install the package in development mode:
```bash
pip install -e .
```

### Dependencies

The repository relies on standard Python scientific packages:
- `numpy` >= 1.21.0
- `pandas` >= 1.3.0
- `scikit-learn` >= 1.0.0
- `matplotlib` >= 3.5.0
- `seaborn` >= 0.12.0
- `faiss-cpu` >= 1.7.0 (for Linux/Windows) or `faiss` >= 1.7.0 (for macOS)

**Note:** The `faiss_example.py` script requires the FAISS library. If it's not available in your environment, the script will print a warning instead of raising an error.

## Usage

### Running Individual Scripts

Each script can be executed directly from the repository root:

```bash
# Matching examples
python matching/psm_example.py
python matching/faiss_example.py

# Uplift modeling examples
python uplift_modeling/uplift_modeling_example.py
python uplift_modeling/s_learner_example.py

# General utilities
python utils/gbdt_example.py

# Model evaluation
python evaluation/model_evaluation.py
```

### Using the Data Ingestion Module

You can import and use the data ingestion functions in your own scripts:

```python
from data import generate_psm_data, generate_uplift_data, load_data, load_data_from_s3

# Generate synthetic data
psm_data = generate_psm_data(n=500, seed=0)
uplift_data = generate_uplift_data(n=1000, seed=42)

# Load data from local file
df = load_data('path/to/your/data.csv')

# Load data from S3 (requires boto3: pip install boto3)
df = load_data_from_s3('my-bucket', 'path/to/data.csv')
```

**📖 For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md)**

## Notes

- Since these scripts create random synthetic data on every run, your numerical results may differ from those shown in the comments, but the overall interpretation should remain the same.
- All scripts use random seeds for reproducibility where applicable.
- The data ingestion module centralizes data generation logic, making it easier to maintain and extend.

## Contributing

This repository is structured to be easily extensible. To add new examples:
1. Place scripts in the appropriate folder (`matching/`, `uplift_modeling/`, or `utils/`)
2. Use the data ingestion module for data generation
3. Update this README with documentation

## License

[Add your license information here]

## Author

Gagan Grewal
