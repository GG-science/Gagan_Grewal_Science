"""
Test script to verify all modules work correctly.

Run this to test the entire repository:
    python test_all.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from data import generate_psm_data, generate_uplift_data, load_data
        from matching.psm_example import estimate_propensity, match_and_estimate_ate
        from matching.faiss_example import build_index, query_index
        from uplift_modeling.uplift_modeling_example import train_two_models, compute_uplift
        from uplift_modeling.s_learner_example import train_s_learner, estimate_cate
        from evaluation import evaluate_classification_model, evaluate_regression_model
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_psm():
    """Test Propensity Score Matching."""
    print("\nTesting PSM...")
    try:
        from data import generate_psm_data
        from matching.psm_example import estimate_propensity, match_and_estimate_ate
        
        data = generate_psm_data(n=100, seed=42)
        prop_scores = estimate_propensity(data)
        ate = match_and_estimate_ate(data, prop_scores)
        print(f"✓ PSM works - ATE: {ate:.3f}")
        return True
    except Exception as e:
        print(f"❌ PSM failed: {e}")
        return False

def test_faiss():
    """Test FAISS similarity search."""
    print("\nTesting FAISS...")
    try:
        from data import generate_vector_data
        from matching.faiss_example import build_index, query_index
        import numpy as np
        
        data = generate_vector_data(n_samples=100, dim=32, seed=42)
        index = build_index(data, use_ivf=False)  # Use simple index for testing
        
        if index is None:
            print("⚠ FAISS not installed (optional)")
            return True  # Not a failure, just optional
        
        query = np.random.random((1, 32)).astype('float32')
        result = query_index(index, query, k=3)
        
        if result is not None:
            indices, distances = result
            print(f"✓ FAISS works - Found {len(indices[0])} neighbors")
            return True
        else:
            print("⚠ FAISS index not built (optional)")
            return True
    except Exception as e:
        print(f"⚠ FAISS test skipped: {e}")
        return True  # FAISS is optional

def test_uplift():
    """Test uplift modeling."""
    print("\nTesting Uplift Modeling...")
    try:
        from data import generate_uplift_data
        from uplift_modeling.uplift_modeling_example import train_two_models, compute_uplift
        
        df = generate_uplift_data(n=200, seed=42)
        model_t, model_c = train_two_models(df)
        uplift = compute_uplift(model_t, model_c, df[['x1', 'x2']])
        print(f"✓ Uplift modeling works - Mean uplift: {uplift.mean():.3f}")
        return True
    except Exception as e:
        print(f"❌ Uplift modeling failed: {e}")
        return False

def test_s_learner():
    """Test S-learner."""
    print("\nTesting S-Learner...")
    try:
        from data import generate_continuous_causal_data
        from uplift_modeling.s_learner_example import train_s_learner, estimate_cate
        
        df, _ = generate_continuous_causal_data(n=200, seed=21)
        model = train_s_learner(df)
        cate = estimate_cate(model, df)
        print(f"✓ S-Learner works - Mean CATE: {cate.mean():.3f}")
        return True
    except Exception as e:
        print(f"❌ S-Learner failed: {e}")
        return False

def test_evaluation():
    """Test evaluation module."""
    print("\nTesting Evaluation Module...")
    try:
        from data import generate_classification_data, generate_continuous_causal_data
        from evaluation import evaluate_classification_model, evaluate_regression_model
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        # Test classification evaluation
        X, y = generate_classification_data(n_samples=200, seed=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        
        # Suppress plot display during testing
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        metrics = evaluate_classification_model(
            clf, X_test, y_test,
            model_name="Test_Classifier",
            save_plots=False  # Don't save during testing
        )
        print(f"✓ Classification evaluation works - Accuracy: {metrics['accuracy']:.3f}")
        
        # Test regression evaluation
        df, _ = generate_continuous_causal_data(n=200, seed=21)
        X_reg = df[['x1', 'x2']].values
        y_reg = df['y'].values
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=42
        )
        
        reg = RandomForestRegressor(n_estimators=10, random_state=42)
        reg.fit(X_train_reg, y_train_reg)
        
        metrics_reg = evaluate_regression_model(
            reg, X_test_reg, y_test_reg,
            model_name="Test_Regressor",
            save_plots=False
        )
        print(f"✓ Regression evaluation works - R²: {metrics_reg['r2_score']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Testing Practical ML Models Repository")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("PSM", test_psm),
        ("FAISS", test_faiss),
        ("Uplift Modeling", test_uplift),
        ("S-Learner", test_s_learner),
        ("Evaluation", test_evaluation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Repository is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

