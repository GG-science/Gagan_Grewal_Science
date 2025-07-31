"""
Demonstration of using FAISS (Facebook AI Similarity Search) to build a vector
index and perform nearest neighbour queries.  FAISS is optimised for large
datasets and high dimensional spaces where brute force search becomes too slow.

When run as a script this file will generate a random dataset of vectors,
build an index, query it with a random test vector and print the closest
vectors.  If the `faiss` module is not installed in your environment the code
will fall back to a message explaining how to install it.
"""

import numpy as np

def build_index(data: np.ndarray, n_list: int = 50):
    """
    Build a FAISS index on the given data.

    Parameters
    ----------
    data : np.ndarray
        A two‑dimensional array of shape (n_samples, n_features).
    n_list : int
        Number of clusters to use for the IVF index.

    Returns
    -------
    index : Optional[faiss.Index]
        A trained FAISS index, or None if faiss is not available.
    """
    try:
        import faiss  # type: ignore
    except ImportError:
        print("faiss is not installed. Please install faiss to run this example.")
        return None

    d = data.shape[1]
    # use an IVF index with flat quantizer for demonstration
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_list)
    index.train(data)
    index.add(data)
    return index

def query_index(index, query: np.ndarray, k: int = 5):
    """
    Query the FAISS index and return the indices of the k nearest neighbours.
    """
    if index is None:
        return None
    distances, indices = index.search(query, k)
    return indices, distances

if __name__ == "__main__":
    # Generate random dataset: 1000 vectors of dimension 64
    np.random.seed(42)
    n_samples, dim = 1000, 64
    data = np.random.random((n_samples, dim)).astype('float32')

    # Build FAISS index
    index = build_index(data)

    # Query with a random vector
    test_vec = np.random.random((1, dim)).astype('float32')
    result = query_index(index, test_vec)
    if result is not None:
        indices, distances = result
        print("Nearest neighbours' indices:", indices[0])
        print("Distances:", distances[0])
