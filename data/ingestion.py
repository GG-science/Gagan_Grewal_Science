"""
Data ingestion module for loading and processing datasets.

This module provides utilities for loading data from various sources,
generating synthetic datasets, and preparing data for machine learning
and causal inference tasks.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from a file (CSV, Excel, Parquet, etc.).

    Parameters
    ----------
    file_path : str
        Path to the data file.
    **kwargs
        Additional arguments passed to pandas read functions.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    
    Example
    -------
    >>> df = load_data('data.csv')
    >>> df = load_data('data.parquet')
    >>> df = load_data('data.xlsx', sheet_name='Sheet1')
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.csv':
        return pd.read_csv(file_path, **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path, **kwargs)
    elif suffix == '.parquet':
        return pd.read_parquet(file_path, **kwargs)
    elif suffix == '.json':
        return pd.read_json(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_data_from_s3(
    bucket_name: str,
    key: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region_name: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load data from an S3 bucket.
    
    This function requires boto3 to be installed. If not available, it will
    raise an ImportError with installation instructions.
    
    AWS credentials can be provided in several ways (in order of precedence):
    1. Function parameters (aws_access_key_id, aws_secret_access_key)
    2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    3. AWS credentials file (~/.aws/credentials)
    4. IAM role (when running on EC2/Lambda)
    
    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket.
    key : str
        S3 object key (path to the file within the bucket).
    aws_access_key_id : str, optional
        AWS access key ID. If not provided, uses default AWS credential chain.
    aws_secret_access_key : str, optional
        AWS secret access key. If not provided, uses default AWS credential chain.
    region_name : str, optional
        AWS region name (e.g., 'us-east-1'). If not provided, uses default.
    **kwargs
        Additional arguments passed to pandas read functions.
    
    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    
    Raises
    ------
    ImportError
        If boto3 is not installed.
    
    Example
    -------
    >>> # Using default AWS credentials
    >>> df = load_data_from_s3('my-bucket', 'data/data.csv')
    >>> 
    >>> # With explicit credentials
    >>> df = load_data_from_s3(
    ...     'my-bucket', 
    ...     'data/data.csv',
    ...     aws_access_key_id='YOUR_KEY',
    ...     aws_secret_access_key='YOUR_SECRET'
    ... )
    >>> 
    >>> # Load Parquet from S3
    >>> df = load_data_from_s3('my-bucket', 'data/data.parquet')
    """
    try:
        import boto3
        from io import BytesIO
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 data loading. "
            "Install it with: pip install boto3"
        )
    
    # Create S3 client
    s3_client_kwargs = {}
    if aws_access_key_id and aws_secret_access_key:
        s3_client_kwargs['aws_access_key_id'] = aws_access_key_id
        s3_client_kwargs['aws_secret_access_key'] = aws_secret_access_key
    if region_name:
        s3_client_kwargs['region_name'] = region_name
    
    s3_client = boto3.client('s3', **s3_client_kwargs)
    
    # Get object from S3
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(f"S3 object not found: s3://{bucket_name}/{key}")
    except Exception as e:
        raise RuntimeError(f"Error accessing S3: {str(e)}")
    
    # Read data into memory
    file_content = response['Body'].read()
    file_like = BytesIO(file_content)
    
    # Determine file type from key extension
    key_lower = key.lower()
    
    if key_lower.endswith('.csv'):
        return pd.read_csv(file_like, **kwargs)
    elif key_lower.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_like, **kwargs)
    elif key_lower.endswith('.parquet'):
        return pd.read_parquet(file_like, **kwargs)
    elif key_lower.endswith('.json'):
        return pd.read_json(file_like, **kwargs)
    else:
        # Try to infer from content or default to CSV
        try:
            return pd.read_csv(file_like, **kwargs)
        except Exception:
            raise ValueError(
                f"Could not determine file format for S3 key: {key}. "
                "Supported formats: .csv, .xlsx, .xls, .parquet, .json"
            )


def generate_psm_data(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """
    Generate synthetic data for Propensity Score Matching.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['x1', 'x2', 'treatment', 'outcome'].
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # probability of treatment depends on x1 and x2
    logits = 0.5 * x1 - 0.3 * x2
    prop = 1 / (1 + np.exp(-logits))
    treatment = rng.binomial(1, prop)
    # outcome depends on x1, x2 and treatment effect of +2
    outcome = 1 + 2 * treatment + 0.5 * x1 - 0.2 * x2 + rng.normal(scale=1.0, size=n)
    data = pd.DataFrame({'x1': x1, 'x2': x2, 'treatment': treatment, 'outcome': outcome})
    return data


def generate_uplift_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data for uplift modeling.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['x1', 'x2', 'treatment', 'y'].
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    # treatment assignment with some dependence on x1
    treat_prop = 1 / (1 + np.exp(-0.5 * x1))
    treatment = rng.binomial(1, treat_prop)
    # outcome probabilities differ by treatment and covariates
    base = -0.5 + 0.5 * x1 + 0.3 * x2
    uplift_effect = 0.8  # treatment effect on log-odds
    logits = base + uplift_effect * treatment
    prob = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, prob)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'treatment': treatment, 'y': y})
    return df


def generate_continuous_causal_data(n: int = 600, seed: int = 21) -> Tuple[pd.DataFrame, float]:
    """
    Generate synthetic continuous outcome data for causal inference.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ['x1', 'x2', 'treatment', 'y'].
    true_effect : float
        The true treatment effect used in simulation.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # treatment assignment probability depends on x1
    prop = 1 / (1 + np.exp(-0.4 * x1))
    treatment = rng.binomial(1, prop)
    true_effect = 3.0  # constant treatment effect
    # outcome is a function of x1, x2 and treatment plus noise
    y = 2 + 0.7 * x1 - 0.5 * x2 + true_effect * treatment + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'treatment': treatment, 'y': y})
    return df, true_effect


def generate_classification_data(n_samples: int = 1000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic classification data.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    """
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=seed
    )
    return X, y


def generate_vector_data(n_samples: int = 1000, dim: int = 64, seed: int = 42) -> np.ndarray:
    """
    Generate random vector data for similarity search.

    Parameters
    ----------
    n_samples : int
        Number of vectors to generate.
    dim : int
        Dimension of each vector.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, dim) with random vectors.
    """
    np.random.seed(seed)
    return np.random.random((n_samples, dim)).astype('float32')


def validate_data(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that a DataFrame contains required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : list
        List of required column names.

    Returns
    -------
    bool
        True if all required columns are present, False otherwise.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def split_data(df: pd.DataFrame, 
               test_size: float = 0.2, 
               random_state: int = 42,
               stratify: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to split.
    test_size : float
        Proportion of data to use for testing.
    random_state : int
        Random seed for reproducibility.
    stratify : str, optional
        Column name to stratify on.

    Returns
    -------
    train_df : pd.DataFrame
        Training data.
    test_df : pd.DataFrame
        Test data.
    """
    from sklearn.model_selection import train_test_split
    
    if stratify is not None:
        stratify_col = df[stratify]
    else:
        stratify_col = None
    
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_col
    )
    return train_df, test_df

