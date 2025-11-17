"""
Demonstration of using FAISS (Facebook AI Similarity Search) to build a vector
index and perform nearest neighbour queries.  FAISS is optimised for large
datasets and high dimensional spaces where brute force search becomes too slow.

When run as a script this file will generate a random dataset of vectors,
build an index, query it with a random test vector and print the closest
vectors.  If the `faiss` module is not installed in your environment the code
will fall back to a message explaining how to install it.
"""

import sys
from pathlib import Path

# Add parent directory to path to import data module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from data import generate_vector_data

def build_index(data: np.ndarray, n_list: int = None, use_ivf: bool = True):
    """
    Build a FAISS index on the given data.

    Parameters
    ----------
    data : np.ndarray
        A two‑dimensional array of shape (n_samples, n_features).
    n_list : int, optional
        Number of clusters to use for the IVF index. If None, auto-calculated.
    use_ivf : bool
        If True, use IndexIVFFlat (faster for large datasets). 
        If False, use IndexFlatL2 (simpler, works for any size).

    Returns
    -------
    index : Optional[faiss.Index]
        A trained FAISS index, or None if faiss is not available.
    """
    try:
        import faiss  # type: ignore
    except ImportError:
        print("faiss is not installed. Please install faiss to run this example.")
        print("Install with: pip install faiss-cpu")
        return None

    d = data.shape[1]
    n_samples = data.shape[0]
    
    # For small datasets or if IVF is disabled, use simple flat index
    if not use_ivf or n_samples < 1000:
        print(f"Using IndexFlatL2 (simple brute-force search) for {n_samples} vectors")
        index = faiss.IndexFlatL2(d)
        index.add(data)
        return index
    
    # For larger datasets, use IVF index
    if n_list is None:
        # Auto-calculate n_list: should be between sqrt(n) and n/4
        n_list = min(max(int(np.sqrt(n_samples)), 10), n_samples // 4)
    
    # Ensure we have enough training data (need at least n_list * 39)
    min_training = n_list * 39
    if n_samples < min_training:
        print(f"Warning: Dataset size ({n_samples}) is small for IVF with {n_list} clusters.")
        print(f"Using IndexFlatL2 instead for better results.")
        index = faiss.IndexFlatL2(d)
        index.add(data)
        return index
    
    print(f"Using IndexIVFFlat with {n_list} clusters for {n_samples} vectors")
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_list)
    
    # Train the index
    index.train(data)
    
    # Add data to index
    index.add(data)
    
    # Set nprobe for querying (number of clusters to search)
    # Higher nprobe = more accurate but slower
    index.nprobe = min(10, n_list)  # Search in up to 10 clusters
    
    return index

def query_index(index, query: np.ndarray, k: int = 5):
    """
    Query the FAISS index and return the indices of the k nearest neighbours.
    
    Parameters
    ----------
    index : faiss.Index
        A trained FAISS index.
    query : np.ndarray
        Query vector(s) of shape (1, n_features) or (n_queries, n_features).
    k : int
        Number of nearest neighbors to return.
    
    Returns
    -------
    tuple or None
        (indices, distances) tuple, or None if index is None.
        - indices: array of shape (n_queries, k) with neighbor indices
        - distances: array of shape (n_queries, k) with L2 distances
    """
    if index is None:
        return None
    
    # Ensure query is 2D and float32
    if query.ndim == 1:
        query = query.reshape(1, -1)
    query = query.astype('float32')
    
    # Ensure k doesn't exceed number of vectors in index
    if hasattr(index, 'ntotal'):
        k = min(k, index.ntotal)
    
    distances, indices = index.search(query, k)
    return indices, distances

if __name__ == "__main__":
    print("="*60)
    print("FAISS Similarity Search Example")
    print("="*60)
    
    # Generate random dataset: 1000 vectors of dimension 64
    print("\n1. Generating vector data...")
    data = generate_vector_data(n_samples=1000, dim=64, seed=42)
    print(f"   Generated {data.shape[0]} vectors of dimension {data.shape[1]}")

    # Build FAISS index
    print("\n2. Building FAISS index...")
    index = build_index(data, use_ivf=True)
    
    if index is None:
        print("\n❌ Failed to build index. Please install FAISS:")
        print("   pip install faiss-cpu")
        exit(1)

    # Query with a random vector
    print("\n3. Querying index...")
    np.random.seed(42)
    dim = 64
    test_vec = np.random.random((1, dim)).astype('float32')
    print(f"   Query vector shape: {test_vec.shape}")
    
    result = query_index(index, test_vec, k=5)
    if result is not None:
        indices, distances = result
        print("\n4. Results:")
        print("   Nearest neighbours' indices:", indices[0])
        print("   L2 Distances:", distances[0])
        print(f"\n✓ Successfully found {len(indices[0])} nearest neighbors!")
    else:
        print("\n❌ Query failed. Index may not be properly initialized.")
