"""
Interactive two-panel Seaborn scatter-plot for PEDRO.
"""
from __future__ import annotations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def fade_red(distances: np.ndarray) -> np.ndarray:
    """Furthest = strongest red; closest = light pink."""
    norm = (distances.max() - distances) / (np.ptp(distances) + 1e-8)
    return np.stack([
        np.ones_like(norm),     
        0.8 * norm,             
        0.8 * norm              
    ], axis=1)

def fade_orange(distances: np.ndarray) -> np.ndarray:
    """Furthest = dark orange; closest = pastel orange."""
    norm = (distances.max() - distances) / (np.ptp(distances) + 1e-8)
    return np.stack([
        np.ones_like(norm),           
        0.6 * norm + 0.2,               
        0.2 * norm                     
    ], axis=1)


def plot_interactive_seaborn(pedro, point_size: int = 25):
    """
    Parameters
    ----------
    pedro : pedro.visualizer.Pedro
        A ready-made Pedro instance (embeddings reduced + KNNs computed).
    point_size : int
        Base marker size.
    """
    sns.set_style("whitegrid")

    coords = pedro.reduced
    n      = len(coords)

    fig, (ax_low, ax_high) = plt.subplots(
        1, 2, figsize=(11, 5), constrained_layout=True
    )
    fig.suptitle("PEDRO – Seaborn interactive KNN viewer", fontweight="bold")

    # Default color = light grey
    grey_rgb = np.full((n, 3), [0.85, 0.85, 0.85])
    scat_low  = ax_low.scatter(coords[:, 0], coords[:, 1],
                               c=grey_rgb, s=point_size, edgecolors="none")
    scat_high = ax_high.scatter(coords[:, 0], coords[:, 1],
                                c=grey_rgb, s=point_size, edgecolors="none")

    ax_low.set_title("Low-dim neighbours")
    ax_high.set_title("High-dim neighbours")

    # ── KD-tree for nearest-point search in 2-D screen coords ────────────
    tree = cKDTree(coords)

    def on_motion(event):
        """Mouse-move callback: updates colours based on nearest point."""
        if event.inaxes not in (ax_low, ax_high):
            return  # outside either subplot

        if event.xdata is None or event.ydata is None:
            return

        _, idx = tree.query([event.xdata, event.ydata], k=1)

        low_sel  = pedro.low_map[idx]
        high_sel = pedro.high_map[idx]

        # ── Base color state ─────────────────────────────────────────────
        colours_low  = np.full((n, 3), [0.85, 0.85, 0.85])
        colours_high = np.full((n, 3), [0.85, 0.85, 0.85])

        # ── Compute distances & fade ─────────────────────────────────────
        low_sel = np.array(sorted(low_sel), dtype=int)
        high_sel = np.array(sorted(high_sel), dtype=int)

        low_coords  = coords[low_sel]
        high_coords = coords[high_sel]

        d_low  = np.linalg.norm(low_coords - coords[idx], axis=1)
        d_high = np.linalg.norm(high_coords - coords[idx], axis=1)

        colours_low[low_sel]   = fade_orange(d_low)
        colours_high[high_sel] = fade_red(d_high)

        colours_low[idx]  = [0.0, 0.0, 0.5]
        colours_high[idx] = [0.0, 0.0, 0.5]

        scat_low.set_facecolors(colours_low)
        scat_high.set_facecolors(colours_high)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    plt.show()
