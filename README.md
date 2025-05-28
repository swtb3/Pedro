<p align="center">
  <img src="assets\pedro_logo.png" alt="PEDRO Logo" width="400">
</p>
<!-- 
<h1 align="center">Pre/Post Embedding Dimension Reduction Observer</h1>
<p align="center"><b></b></p>
<p align="center">
  Visualize how dimensionality reduction alters neighborhood structure in embedding space.
</p> -->

<p align="center">
  <a href="https://pypi.org/project/pedro/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pedro?color=blue"></a>
  <a href="https://github.com/yourusername/pedro/actions"><img alt="Build Status" src="https://img.shields.io/github/actions/workflow/status/yourusername/pedro/ci.yml?branch=main"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
  <a href="https://python.org"><img alt="Python" src="https://img.shields.io/badge/python-3.8%2B-blue"></a>
</p>

---

**P**re/**P**ost **E**mbedding **D**imension **R**eduction **O**bserver (**PEDRO**) is a python library that allows for side by side comparison between the clustering of embeddings in their native (high dimensional space) and their low dimension projections (as provided by a dimensionality reduction algorithm).

It currently supports the following backends to produce dr-embeddings (dimension reduced embeddings):
- **PacMap**
- **UMAP**
- **t-SNE**

Alternatively, the user can supply their own precomputed dr-embeddings alongside their own embeddings. Kwargs can be passed to provide input to the backends.

The result is an interactive plot (provided by plotly), hovering over points in the left pane reveals top-k neightbours in the low dimension space; and the right for the high dimensional space.
