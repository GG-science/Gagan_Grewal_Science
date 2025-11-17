# Execution Structure Guide

This document shows the step-by-step execution flow for each module in the repository.

## Matching Techniques

### Propensity Score Matching (`matching/psm_example.py`)

```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Data Preparation                                │
│ ─────────────────────────────────────────────────────── │
│ Option A: Generate synthetic data                       │
│   data = generate_psm_data(n=500, seed=0)              │
│                                                          │
│ Option B: Load from file                                │
│   data = load_data('your_data.csv')                     │
│                                                          │
│ Option C: Load from S3                                  │
│   data = load_data_from_s3('bucket', 'key')            │
│                                                          │
│ Required columns: ['x1', 'x2', 'treatment', 'outcome'] │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Estimate Propensity Scores                     │
│ ─────────────────────────────────────────────────────── │
│ prop_scores = estimate_propensity(data)                │
│                                                          │
│ Internal process:                                       │
│   1. Extract features: data[['x1', 'x2']]              │
│   2. Extract treatment: data['treatment']              │
│   3. Fit LogisticRegression model                       │
│   4. Predict probabilities → propensity scores         │
│                                                          │
│ Returns: np.ndarray of propensity scores                │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Match and Estimate ATE                         │
│ ─────────────────────────────────────────────────────── │
│ ate = match_and_estimate_ate(data, prop_scores)       │
│                                                          │
│ Internal process:                                       │
│   1. Separate treated (treatment==1) and control (==0)  │
│   2. Build NearestNeighbors on control propensity      │
│   3. Match each treated unit to nearest control         │
│   4. Calculate effect = treated_outcome - control_outcome│
│   5. Average effects → Average Treatment Effect (ATE)   │
│                                                          │
│ Returns: float (Average Treatment Effect)              │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Output Results                                 │
│ ─────────────────────────────────────────────────────── │
│ print(f"Estimated ATE (PSM): {ate:.3f}")               │
└─────────────────────────────────────────────────────────┘
```

**Function Signatures:**
```python
def estimate_propensity(data: pd.DataFrame) -> np.ndarray:
    """Returns propensity scores for each observation."""
    
def match_and_estimate_ate(data: pd.DataFrame, prop_scores: np.ndarray) -> float:
    """Returns Average Treatment Effect."""
```

---

### FAISS Similarity Search (`matching/faiss_example.py`)

```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Prepare Vector Data                             │
│ ─────────────────────────────────────────────────────── │
│ Option A: Generate synthetic vectors                    │
│   data = generate_vector_data(n_samples=1000, dim=64)  │
│                                                          │
│ Option B: Load your own vectors                         │
│   df = load_data('vectors.csv')                         │
│   data = df.values.astype('float32')                    │
│                                                          │
│ Shape: (n_samples, n_dimensions)                        │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Build FAISS Index                              │
│ ─────────────────────────────────────────────────────── │
│ index = build_index(data, n_list=50)                   │
│                                                          │
│ Internal process:                                       │
│   1. Create IndexIVFFlat with flat quantizer            │
│   2. Train index on data                                │
│   3. Add all vectors to index                           │
│                                                          │
│ Returns: faiss.Index object (or None if FAISS missing) │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Query Index                                    │
│ ─────────────────────────────────────────────────────── │
│ query_vec = np.array([[0.1, 0.2, ...]]).astype('float32')│
│ indices, distances = query_index(index, query_vec, k=5) │
│                                                          │
│ Internal process:                                       │
│   1. Search for k nearest neighbors                     │
│   2. Return indices and distances                       │
│                                                          │
│ Returns: (indices, distances) tuple                     │
└─────────────────────────────────────────────────────────┘
```

**Function Signatures:**
```python
def build_index(data: np.ndarray, n_list: int = 50) -> Optional[faiss.Index]:
    """Builds and returns a trained FAISS index."""
    
def query_index(index, query: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (indices, distances) of k nearest neighbors."""
```

---

## Uplift Modeling Techniques

### Two-Model Uplift (`uplift_modeling/uplift_modeling_example.py`)

```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Data Preparation                                │
│ ─────────────────────────────────────────────────────── │
│ Option A: Generate synthetic data                       │
│   df = generate_uplift_data(n=1000, seed=42)           │
│                                                          │
│ Option B: Load your data                               │
│   df = load_data('your_data.csv')                       │
│                                                          │
│ Required columns: ['x1', 'x2', 'treatment', 'y']       │
│   - treatment: binary (0 or 1)                          │
│   - y: binary outcome (0 or 1)                          │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Train Two Models                               │
│ ─────────────────────────────────────────────────────── │
│ model_t, model_c = train_two_models(df)                │
│                                                          │
│ Internal process:                                       │
│   1. Split data: treated (treatment==1) vs control (==0)│
│   2. Train LogisticRegression on treated group         │
│      → model_t (treatment model)                        │
│   3. Train LogisticRegression on control group         │
│      → model_c (control model)                          │
│                                                          │
│ Returns: (model_t, model_c) tuple                      │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Compute Uplift                                 │
│ ─────────────────────────────────────────────────────── │
│ features = df[['x1', 'x2']]                            │
│ uplift = compute_uplift(model_t, model_c, features)    │
│                                                          │
│ Internal process:                                       │
│   1. Predict P(y=1 | treatment) using model_t           │
│   2. Predict P(y=1 | control) using model_c            │
│   3. Calculate: uplift = P(treatment) - P(control)     │
│                                                          │
│ Returns: np.ndarray of uplift scores                    │
└─────────────────────────────────────────────────────────┘
```

**Function Signatures:**
```python
def train_two_models(df: pd.DataFrame) -> Tuple[LogisticRegression, LogisticRegression]:
    """Returns (treatment_model, control_model)."""
    
def compute_uplift(model_t, model_c, X: pd.DataFrame) -> np.ndarray:
    """Returns uplift scores for each observation."""
```

---

### S-Learner (`uplift_modeling/s_learner_example.py`)

```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Data Preparation                                │
│ ─────────────────────────────────────────────────────── │
│ Option A: Generate synthetic data                      │
│   df, true_effect = generate_continuous_causal_data()  │
│                                                          │
│ Option B: Load your data                                │
│   df = load_data('your_data.csv')                       │
│                                                          │
│ Required columns: ['x1', 'x2', 'treatment', 'y']      │
│   - treatment: binary (0 or 1)                          │
│   - y: continuous outcome                               │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Train S-Learner                                │
│ ─────────────────────────────────────────────────────── │
│ model = train_s_learner(df)                            │
│                                                          │
│ Internal process:                                       │
│   1. Prepare features: df[['x1', 'x2', 'treatment']]  │
│      (treatment is included as a feature!)               │
│   2. Extract outcome: df['y']                           │
│   3. Train GradientBoostingRegressor                    │
│                                                          │
│ Returns: Trained GradientBoostingRegressor             │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Estimate CATE                                   │
│ ─────────────────────────────────────────────────────── │
│ cate = estimate_cate(model, df)                        │
│                                                          │
│ Internal process:                                       │
│   1. Create X0: features with treatment=0             │
│   2. Create X1: features with treatment=1              │
│   3. Predict y0 = model.predict(X0)                    │
│   4. Predict y1 = model.predict(X1)                    │
│   5. Calculate: CATE = y1 - y0                         │
│      (Conditional Average Treatment Effect)             │
│                                                          │
│ Returns: np.ndarray of CATE values                      │
└─────────────────────────────────────────────────────────┘
```

**Function Signatures:**
```python
def train_s_learner(df: pd.DataFrame) -> GradientBoostingRegressor:
    """Trains model with treatment as a feature."""
    
def estimate_cate(model, df: pd.DataFrame) -> np.ndarray:
    """Returns Conditional Average Treatment Effects."""
```

---

## Complete Workflow Example

### End-to-End PSM Pipeline

```python
# 1. Import modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import load_data_from_s3, validate_data
from matching.psm_example import estimate_propensity, match_and_estimate_ate

# 2. Load data
df = load_data_from_s3('my-bucket', 'experiments/data.csv')

# 3. Validate
validate_data(df, ['x1', 'x2', 'treatment', 'outcome'])

# 4. Execute PSM pipeline
prop_scores = estimate_propensity(df)        # Step 2
ate = match_and_estimate_ate(df, prop_scores)  # Step 3

# 5. Results
print(f"ATE: {ate:.3f}")
```

### End-to-End Uplift Modeling Pipeline

```python
# 1. Import modules
from data import load_data, split_data
from uplift_modeling.uplift_modeling_example import train_two_models, compute_uplift
from evaluation import evaluate_classification_model

# 2. Load and split data
df = load_data('uplift_data.csv')
train_df, test_df = split_data(df, test_size=0.2, stratify='treatment')

# 3. Train models
model_t, model_c = train_two_models(train_df)

# 4. Compute uplift
uplift = compute_uplift(model_t, model_c, test_df[['x1', 'x2']])

# 5. Evaluate (optional)
X_test = test_df[test_df['treatment'] == 1][['x1', 'x2']].values
y_test = test_df[test_df['treatment'] == 1]['y'].values
evaluate_classification_model(model_t, X_test, y_test, "Treatment Model")
```

---

## Key Points

1. **All modules import from `data/`**: Centralized data handling
2. **Consistent structure**: Load → Process → Output
3. **Flexible data sources**: Local files, S3, or synthetic data
4. **Validation**: Use `validate_data()` before processing
5. **Evaluation**: Use `evaluation/` module to assess models

For more details, see [USAGE_GUIDE.md](USAGE_GUIDE.md).

