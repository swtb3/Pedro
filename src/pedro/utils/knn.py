from __future__ import annotations
import numpy as np
import warnings


def _hnsw_knn(arr: np.ndarray, k: int) -> np.ndarray:
    import hnswlib

    n, dim = arr.shape
    index  = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n, ef_construction=200, M=16)
    index.add_items(arr, ids=np.arange(n, dtype=np.int64))
    index.set_ef(max(50, 2 * k))
    idx, _ = index.knn_query(arr, k=k)
    return idx.astype("uint32")

def _sk_knn(arr: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    idx = (
        NearestNeighbors(n_neighbors=k, algorithm="auto")
        .fit(arr)
        .kneighbors(arr, return_distance=False)
    )
    return idx.astype("uint32")

def build_knn_indices(arr: np.ndarray, k: int) -> np.ndarray:
    arr32 = arr.astype("float32", copy=False)

    try:
        return _hnsw_knn(arr32, k)
    except ModuleNotFoundError:
        warnings.warn("hnswlib not installed – falling back to sklearn.", RuntimeWarning)
    except Exception as e:
        warnings.warn(f"hnswlib failed ({e!s}) – falling back to sklearn.", RuntimeWarning)

    return _sk_knn(arr32, k)
