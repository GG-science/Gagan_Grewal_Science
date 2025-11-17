# Practical ML Models

A practical repository for **causal inference**, **uplift modeling**, and **similarity search** with ready-to-use examples, evaluation tools, and visualizations.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/GG-science/practical_ml_models.git
cd practical_ml_models
pip install -r requirements.txt

# 2. Run your first example
python matching/psm_example.py
python evaluation/model_evaluation.py  # See evaluation plots
```

## When to Use Each Technique

### 🎯 **Propensity Score Matching (PSM)**
**Use when:** You have observational data and want to estimate the causal effect of a treatment.

**Example scenarios:**
- Did a marketing campaign increase sales?
- Does a new feature improve user engagement?
- What's the impact of a policy change?

**Output:** Average Treatment Effect (ATE) - a single number showing the treatment impact.

```bash
python matching/psm_example.py
# Output: Estimated ATE (PSM): 2.045
```

### 📈 **Uplift Modeling**
**Use when:** You want to identify which individuals will respond best to treatment.

**Two approaches:**

1. **Two-Model Approach** - Simple, interpretable
   ```bash
   python uplift_modeling/uplift_modeling_example.py
   # Output: Uplift scores for each individual
   ```

2. **S-Learner** - More flexible, handles continuous outcomes
   ```bash
   python uplift_modeling/s_learner_example.py
   # Output: Conditional Average Treatment Effects (CATE) per individual
   ```

**Example scenarios:**
- Who should receive a discount coupon?
- Which customers will respond to an email campaign?
- Personalized treatment recommendations

### 🔍 **FAISS Similarity Search**
**Use when:** You need fast nearest neighbor search on large vector datasets.

**Example scenarios:**
- Finding similar products (embeddings)
- Recommendation systems
- Image/video similarity search

```bash
python matching/faiss_example.py
# Output: Nearest neighbor indices and distances
```

## Evaluation & Visualizations

All models can be evaluated with automatic plot generation:

```python
from evaluation import evaluate_classification_model, evaluate_regression_model

# Evaluate any scikit-learn model
metrics = evaluate_classification_model(
    model, X_test, y_test,
    model_name="My Model",
    save_plots=True  # Saves to evaluation/plots/
)
```

### What You Get:

**Classification Models:**
- 📊 Confusion matrix heatmap
- 📈 ROC curve with AUC score
- 📋 Metrics summary (accuracy, precision, recall, F1)

**Regression Models:**
- 📊 Predicted vs Actual scatter plot
- 📈 Residual plot
- 📋 Metrics summary (R², RMSE, MAE, MSE)

**Example output:**
```bash
python evaluation/model_evaluation.py
# Generates plots in evaluation/plots/
# - random_forest_classifier_classification.png
# - random_forest_regressor_regression.png
# - model_comparison_classification.png
```

## Repository Structure

```
practical_ml_models/
├── data/              # Data loading (local files, S3, synthetic data)
├── matching/          # PSM and FAISS similarity search
├── uplift_modeling/   # Two-model and S-learner approaches
├── evaluation/        # Model evaluation with plots
└── utils/             # General ML utilities
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install for S3 support
pip install boto3

# Optional: Install FAISS (for similarity search)
pip install faiss-cpu  # Linux/Windows
# or
pip install faiss      # macOS
```

## Quick Examples

### 1. Estimate Treatment Effect (PSM)

```python
from data import generate_psm_data
from matching.psm_example import estimate_propensity, match_and_estimate_ate

# Load or generate data
data = generate_psm_data(n=500, seed=0)

# Run PSM
prop_scores = estimate_propensity(data)
ate = match_and_estimate_ate(data, prop_scores)
print(f"Average Treatment Effect: {ate:.3f}")
```

### 2. Find Who Benefits Most (Uplift)

```python
from data import generate_uplift_data
from uplift_modeling.uplift_modeling_example import train_two_models, compute_uplift

# Load data
df = generate_uplift_data(n=1000, seed=42)

# Train models
model_t, model_c = train_two_models(df)

# Compute uplift for each person
uplift = compute_uplift(model_t, model_c, df[['x1', 'x2']])
df['uplift'] = uplift

# Target top 10% with highest uplift
top_targets = df.nlargest(len(df)//10, 'uplift')
```

### 3. Evaluate Your Model

```python
from evaluation import evaluate_classification_model
from sklearn.ensemble import RandomForestClassifier

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate with plots
metrics = evaluate_classification_model(
    model, X_test, y_test,
    model_name="Random Forest",
    save_plots=True
)
# Check evaluation/plots/random_forest_classification.png
```

### 4. Use Your Own Data

```python
from data import load_data, validate_data

# Load from file
df = load_data('your_data.csv')

# Or from S3
from data import load_data_from_s3
df = load_data_from_s3('my-bucket', 'data/data.csv')

# Validate required columns
validate_data(df, ['x1', 'x2', 'treatment', 'outcome'])
```

## Testing & Verification

Run all examples to verify everything works:

```bash
# Test matching techniques
python matching/psm_example.py
python matching/faiss_example.py

# Test uplift modeling
python uplift_modeling/uplift_modeling_example.py
python uplift_modeling/s_learner_example.py

# Test evaluation (generates example plots)
python evaluation/model_evaluation.py

# Test utilities
python utils/gbdt_example.py
```

Expected outputs:
- ✅ All scripts run without errors
- ✅ Plots saved to `evaluation/plots/` directory
- ✅ Metrics printed to console

## Understanding the Outputs

### PSM Output
```
Estimated ATE (PSM): 2.045
```
- **ATE > 0**: Treatment has positive effect
- **ATE < 0**: Treatment has negative effect
- **Magnitude**: Size of the effect

### Uplift Output
```
Uplift scores (first 10): [0.15, 0.08, -0.02, 0.23, ...]
```
- **Positive**: Person benefits from treatment
- **Negative**: Person may be harmed by treatment
- **Higher = Better**: Target those with highest uplift

### Evaluation Plots
- **Confusion Matrix**: Shows prediction accuracy by class
- **ROC Curve**: Trade-off between true positive and false positive rates
- **Residual Plot**: For regression - shows prediction errors
- **Predicted vs Actual**: How well predictions match reality

## Key Concepts Explained

### Propensity Score Matching
**Problem:** In observational data, treated and control groups may differ, causing bias.

**Solution:** Match each treated unit to a similar control unit based on propensity score (probability of receiving treatment).

**Result:** Unbiased estimate of treatment effect.

### Uplift Modeling
**Problem:** Not everyone responds the same to treatment. Some benefit, some don't.

**Solution:** Model the difference in outcomes between treatment and control for each individual.

**Result:** Identify who will benefit most from treatment.

### S-Learner vs Two-Model
- **Two-Model**: Train separate models on treated/control groups. Simple, interpretable.
- **S-Learner**: Single model with treatment as a feature. More flexible, can handle interactions.

## Data Requirements

| Technique | Required Columns | Outcome Type |
|-----------|-----------------|--------------|
| PSM | Features + `treatment` + `outcome` | Continuous |
| Two-Model Uplift | Features + `treatment` + `y` | Binary |
| S-Learner | Features + `treatment` + `y` | Continuous |

## Next Steps

1. **Start with examples**: Run the scripts to see how they work
2. **Use your data**: Replace synthetic data with your own
3. **Evaluate results**: Use the evaluation module to assess models
4. **Read detailed guides**: See [USAGE_GUIDE.md](USAGE_GUIDE.md) and [EXECUTION_STRUCTURE.md](EXECUTION_STRUCTURE.md)

## Troubleshooting

**Import errors?**
- Make sure you're running from the repository root
- Install all dependencies: `pip install -r requirements.txt`

**FAISS not working?**
- Install: `pip install faiss-cpu` (or `faiss` on macOS)
- Script will use simpler index if FAISS unavailable

**Plot not showing?**
- Check `evaluation/plots/` directory
- Ensure matplotlib backend is configured

## Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Detailed usage with S3, custom data, workflows
- **[EXECUTION_STRUCTURE.md](EXECUTION_STRUCTURE.md)** - Step-by-step execution flows

## License

[Add your license information here]

## Author

Gagan Grewal
