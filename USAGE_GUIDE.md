# Usage Guide

This guide explains how to use this repository, how all the files connect, and how to plug in your own datasets.

## Table of Contents
1. [How Files Are Connected](#how-files-are-connected)
2. [Using Your Own Dataset](#using-your-own-dataset)
3. [Execution Structure](#execution-structure)
4. [Complete Workflow Examples](#complete-workflow-examples)

---

## How Files Are Connected

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    data/ingestion.py                         │
│  (Central Data Hub - All modules import from here)         │
│  • load_data() - Load from files/S3                         │
│  • generate_*_data() - Generate synthetic data              │
│  • validate_data() - Validate data structure                │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ imports
                            ▼
        ┌───────────────────┴───────────────────┐
        │                                       │
        ▼                                       ▼
┌───────────────┐                    ┌──────────────────┐
│  matching/    │                    │ uplift_modeling/ │
│  • psm_example│                    │  • uplift_*      │
│  • faiss_*    │                    │  • s_learner_*   │
└───────────────┘                    └──────────────────┘
        │                                       │
        │                                       │
        └───────────────┬───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   evaluation/         │
            │   model_evaluation.py │
            │   (Evaluates models)  │
            └───────────────────────┘
```

### Import Pattern

All modules follow this pattern to connect to the data module:

```python
import sys
from pathlib import Path

# Add parent directory to path to import data module
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import generate_psm_data, load_data  # Import what you need
```

**Why this pattern?**
- Each script can be run independently from the repository root
- No need to install the package (though you can with `pip install -e .`)
- Works with relative imports across the directory structure

---

## Using Your Own Dataset

### Option 1: Load from Local File

```python
from data import load_data

# Load CSV
df = load_data('path/to/your/data.csv')

# Load with pandas options
df = load_data('data.csv', sep=';', encoding='utf-8')

# Load Parquet
df = load_data('data.parquet')

# Load Excel
df = load_data('data.xlsx', sheet_name='Sheet1')
```

### Option 2: Load from S3

The data ingestion module now supports S3! Here's how:

```python
from data import load_data_from_s3

# Load from S3 (requires boto3)
df = load_data_from_s3(
    bucket_name='my-bucket',
    key='path/to/data.csv',
    aws_access_key_id='YOUR_KEY',  # Optional if using AWS credentials
    aws_secret_access_key='YOUR_SECRET'  # Optional if using AWS credentials
)

# Or use environment variables or AWS credentials file
df = load_data_from_s3('my-bucket', 'data/data.csv')
```

### Option 3: Use Mock/Synthetic Data

```python
from data import (
    generate_psm_data,
    generate_uplift_data,
    generate_continuous_causal_data,
    generate_classification_data
)

# For Propensity Score Matching
psm_data = generate_psm_data(n=1000, seed=42)

# For Uplift Modeling
uplift_data = generate_uplift_data(n=2000, seed=42)

# For Causal Inference (S-learner)
causal_data, true_effect = generate_continuous_causal_data(n=600, seed=21)

# For Classification
X, y = generate_classification_data(n_samples=1000, seed=42)
```

### Option 4: Use Your Own DataFrame

If you already have a pandas DataFrame, just ensure it has the right columns:

**For PSM:**
```python
# Required columns: ['x1', 'x2', 'treatment', 'outcome']
# OR any feature columns + 'treatment' + 'outcome'
df = pd.DataFrame({
    'feature1': [...],
    'feature2': [...],
    'treatment': [0, 1, 0, 1, ...],  # Binary treatment
    'outcome': [10.5, 12.3, 9.8, ...]  # Continuous outcome
})
```

**For Uplift Modeling:**
```python
# Required columns: ['x1', 'x2', 'treatment', 'y']
# OR any feature columns + 'treatment' + 'y'
df = pd.DataFrame({
    'feature1': [...],
    'feature2': [...],
    'treatment': [0, 1, 0, 1, ...],  # Binary treatment
    'y': [0, 1, 0, 1, ...]  # Binary outcome
})
```

---

## Execution Structure

### Matching Techniques

#### Propensity Score Matching (`matching/psm_example.py`)

**Execution Flow:**
```
1. Load/Generate Data
   └─> generate_psm_data() or load_data()
   
2. Estimate Propensity Scores
   └─> estimate_propensity(data)
       └─> LogisticRegression.fit()
       └─> Returns: propensity scores array
   
3. Match and Estimate ATE
   └─> match_and_estimate_ate(data, prop_scores)
       └─> NearestNeighbors matching
       └─> Returns: Average Treatment Effect (ATE)
   
4. Output Results
   └─> Print ATE
```

**Using with your data:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import load_data, validate_data
from matching.psm_example import estimate_propensity, match_and_estimate_ate

# Load your data
df = load_data('your_data.csv')
# OR use S3: df = load_data_from_s3('bucket', 'key')

# Validate required columns
validate_data(df, ['x1', 'x2', 'treatment', 'outcome'])
# OR if your columns have different names:
# df = df.rename(columns={'old_name': 'x1', 'another': 'x2'})

# Run PSM
prop_scores = estimate_propensity(df)
ate = match_and_estimate_ate(df, prop_scores)
print(f"Estimated ATE: {ate:.3f}")
```

#### FAISS Similarity Search (`matching/faiss_example.py`)

**Execution Flow:**
```
1. Generate/Load Vector Data
   └─> generate_vector_data() or your own vectors
   
2. Build FAISS Index
   └─> build_index(data)
       └─> Creates IndexIVFFlat
       └─> Trains and adds vectors
   
3. Query Index
   └─> query_index(index, query_vector)
       └─> Returns: nearest neighbor indices and distances
```

**Using with your data:**
```python
from data import load_data
from matching.faiss_example import build_index, query_index
import numpy as np

# Load your vector data (each row is a vector)
df = load_data('vectors.csv')
vectors = df.values.astype('float32')

# Build index
index = build_index(vectors, n_list=50)

# Query
query_vec = np.array([[0.1, 0.2, 0.3, ...]]).astype('float32')
indices, distances = query_index(index, query_vec, k=5)
```

### Uplift Modeling Techniques

#### Two-Model Uplift (`uplift_modeling/uplift_modeling_example.py`)

**Execution Flow:**
```
1. Load/Generate Data
   └─> generate_uplift_data() or load_data()
   
2. Train Two Models
   └─> train_two_models(df)
       ├─> Train model on treated group (treatment == 1)
       └─> Train model on control group (treatment == 0)
   
3. Compute Uplift
   └─> compute_uplift(model_t, model_c, X)
       ├─> Predict probability with treatment model
       ├─> Predict probability with control model
       └─> Returns: uplift = P(treatment) - P(control)
```

**Using with your data:**
```python
from data import load_data, validate_data
from uplift_modeling.uplift_modeling_example import train_two_models, compute_uplift

# Load your data
df = load_data('your_data.csv')
validate_data(df, ['x1', 'x2', 'treatment', 'y'])

# Train models
model_t, model_c = train_two_models(df)

# Compute uplift for all samples
features = df[['x1', 'x2']]
uplift_scores = compute_uplift(model_t, model_c, features)

# Add to dataframe
df['uplift'] = uplift_scores
```

#### S-Learner (`uplift_modeling/s_learner_example.py`)

**Execution Flow:**
```
1. Load/Generate Data
   └─> generate_continuous_causal_data() or load_data()
   
2. Train S-Learner
   └─> train_s_learner(df)
       └─> GradientBoostingRegressor with treatment as feature
   
3. Estimate CATE
   └─> estimate_cate(model, df)
       ├─> Predict with treatment=0
       ├─> Predict with treatment=1
       └─> Returns: CATE = prediction(treatment=1) - prediction(treatment=0)
```

**Using with your data:**
```python
from data import load_data
from uplift_modeling.s_learner_example import train_s_learner, estimate_cate

# Load your data
df = load_data('your_data.csv')

# Train model
model = train_s_learner(df)

# Estimate CATE for each observation
cate = estimate_cate(model, df)
df['cate'] = cate
```

---

## Complete Workflow Examples

### Example 1: End-to-End PSM with S3 Data

```python
"""
Complete workflow: Load from S3 → Run PSM → Evaluate Results
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data import load_data_from_s3, validate_data
from matching.psm_example import estimate_propensity, match_and_estimate_ate

# 1. Load data from S3
print("Loading data from S3...")
df = load_data_from_s3(
    bucket_name='my-data-bucket',
    key='experiments/treatment_data.csv'
)

# 2. Validate and prepare data
print("Validating data...")
validate_data(df, ['x1', 'x2', 'treatment', 'outcome'])

# 3. Run PSM
print("Estimating propensity scores...")
prop_scores = estimate_propensity(df)

print("Matching and estimating ATE...")
ate = match_and_estimate_ate(df, prop_scores)

# 4. Results
print(f"\n{'='*60}")
print(f"Propensity Score Matching Results")
print(f"{'='*60}")
print(f"Estimated Average Treatment Effect: {ate:.3f}")
print(f"Sample size: {len(df)}")
print(f"Treated: {df['treatment'].sum()}")
print(f"Control: {(df['treatment'] == 0).sum()}")
```

### Example 2: Uplift Modeling with Local Data + Evaluation

```python
"""
Complete workflow: Load local data → Train uplift model → Evaluate
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data import load_data, split_data
from uplift_modeling.uplift_modeling_example import train_two_models, compute_uplift
from evaluation import evaluate_classification_model
from sklearn.ensemble import RandomForestClassifier

# 1. Load your data
df = load_data('my_uplift_data.csv')

# 2. Split data
train_df, test_df = split_data(df, test_size=0.2, stratify='treatment')

# 3. Train uplift models
print("Training uplift models...")
model_t, model_c = train_two_models(train_df)

# 4. Compute uplift on test set
test_features = test_df[['x1', 'x2']]
uplift = compute_uplift(model_t, model_c, test_features)
test_df['uplift'] = uplift

# 5. Evaluate treatment model (optional)
# Create a combined model for evaluation
X_test = test_df[test_df['treatment'] == 1][['x1', 'x2']].values
y_test = test_df[test_df['treatment'] == 1]['y'].values

evaluate_classification_model(
    model_t, X_test, y_test,
    model_name="Uplift Treatment Model"
)

# 6. Results
print(f"\nUplift Statistics:")
print(f"Mean uplift: {uplift.mean():.3f}")
print(f"Max uplift: {uplift.max():.3f}")
print(f"Min uplift: {uplift.min():.3f}")
```

### Example 3: Using Mock Data for Quick Testing

```python
"""
Quick test with synthetic data - no external files needed
"""
from data import generate_psm_data, generate_uplift_data
from matching.psm_example import estimate_propensity, match_and_estimate_ate
from uplift_modeling.uplift_modeling_example import train_two_models, compute_uplift

# Test PSM with mock data
print("Testing PSM with synthetic data...")
psm_data = generate_psm_data(n=500, seed=42)
prop_scores = estimate_propensity(psm_data)
ate = match_and_estimate_ate(psm_data, prop_scores)
print(f"PSM ATE: {ate:.3f}")

# Test Uplift with mock data
print("\nTesting Uplift Modeling with synthetic data...")
uplift_data = generate_uplift_data(n=1000, seed=42)
model_t, model_c = train_two_models(uplift_data)
uplift = compute_uplift(model_t, model_c, uplift_data[['x1', 'x2']])
print(f"Mean uplift: {uplift.mean():.3f}")
```

---

## Data Format Requirements

### For Propensity Score Matching

**Required columns:**
- Feature columns (e.g., `x1`, `x2`, or any feature names)
- `treatment`: Binary (0 or 1)
- `outcome`: Continuous numeric values

**Example:**
```python
df = pd.DataFrame({
    'age': [25, 30, 35, ...],
    'income': [50000, 60000, 70000, ...],
    'treatment': [0, 1, 0, 1, ...],
    'outcome': [10.5, 12.3, 9.8, 11.2, ...]
})
```

### For Uplift Modeling

**Required columns:**
- Feature columns (e.g., `x1`, `x2`)
- `treatment`: Binary (0 or 1)
- `y`: Binary outcome (0 or 1)

**Example:**
```python
df = pd.DataFrame({
    'feature1': [...],
    'feature2': [...],
    'treatment': [0, 1, 0, 1, ...],
    'y': [0, 1, 0, 1, ...]  # Binary outcome
})
```

### For S-Learner

**Required columns:**
- Feature columns
- `treatment`: Binary (0 or 1)
- `y`: Continuous outcome

**Example:**
```python
df = pd.DataFrame({
    'x1': [...],
    'x2': [...],
    'treatment': [0, 1, 0, 1, ...],
    'y': [10.5, 12.3, 9.8, 11.2, ...]  # Continuous
})
```

---

## Tips and Best Practices

1. **Always validate your data** before running analysis:
   ```python
   from data import validate_data
   validate_data(df, required_columns=['x1', 'x2', 'treatment', 'outcome'])
   ```

2. **Use seeds for reproducibility** when generating synthetic data:
   ```python
   data = generate_psm_data(n=1000, seed=42)  # Same seed = same data
   ```

3. **Split your data** for proper evaluation:
   ```python
   from data import split_data
   train_df, test_df = split_data(df, test_size=0.2, stratify='treatment')
   ```

4. **Save your results**:
   ```python
   df['propensity_score'] = prop_scores
   df['uplift'] = uplift_scores
   df.to_csv('results.csv', index=False)
   ```

5. **Use evaluation module** to assess model performance:
   ```python
   from evaluation import evaluate_classification_model
   metrics = evaluate_classification_model(model, X_test, y_test)
   ```

---

## Troubleshooting

**Problem:** Import errors when running scripts
- **Solution:** Always run scripts from the repository root directory

**Problem:** Column name mismatches
- **Solution:** Rename your columns to match expected names, or modify the scripts to use your column names

**Problem:** S3 access denied
- **Solution:** Check AWS credentials (environment variables, ~/.aws/credentials, or IAM role)

**Problem:** Data format issues
- **Solution:** Use `validate_data()` to check required columns before processing

---

## Next Steps

1. Start with the examples using synthetic data to understand the flow
2. Adapt the examples to your data format
3. Use the evaluation module to assess your models
4. Extend the codebase with your own custom functions

For questions or issues, refer to the individual script docstrings or the main README.md.

