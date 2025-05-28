from __future__ import annotations
import numpy as np
from .utils.knn import build_knn_indices
from .backends import umap_reduce, pacmap_reduce, tsne_reduce
from .utils.plotting import plot_interactive_seaborn

class Pedro:
    """
    1. Store high-dim embeddings
    2. Reduce to 2-D (if coords not supplied)
    3. Build fast k-NN look-ups (hnswlib → sklearn fallback)
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        k: int = 10,
        reduced_coords: np.ndarray | None = None,
        method: str = "umap",
        dr_kwargs: dict | None = None,
        force_float32: bool = True,
    ):
        self.embeddings = (
            embeddings.astype("float32") if force_float32 else np.asarray(embeddings)
        )
        self.k       = k
        self.kwargs  = dr_kwargs or {}

        self.reduced = (
            np.asarray(reduced_coords, dtype="float32")
            if reduced_coords is not None
            else self._reduce(method)
        )

        self.low_knn  = build_knn_indices(self.reduced, k)
        self.high_knn = build_knn_indices(self.embeddings, k)

        self.low_map  = {i: set(row) | {i} for i, row in enumerate(self.low_knn)}
        self.high_map = {i: set(row) | {i} for i, row in enumerate(self.high_knn)}

    def _reduce(self, method: str) -> np.ndarray:
        if method == "umap":
            return umap_reduce(self.embeddings, **self.kwargs)
        if method == "pacmap":
            return pacmap_reduce(self.embeddings, **self.kwargs)
        if method == "tsne":
            return tsne_reduce(self.embeddings, **self.kwargs)
        raise ValueError(f"Unknown DR method {method!r}")
    
    def plot(self):
        plot_interactive_seaborn(self)
