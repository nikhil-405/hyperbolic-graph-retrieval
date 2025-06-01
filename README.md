# Hyperbolic Graph Retrieval

**Scalable and Efficient Graph Retrieval Using Hyperbolic Embeddings and Deep Hashing**

---

## Overview

Modern applications require rapid and scalable retrieval of complex graph-structured data, from social networks to knowledge graphs. **Hyperbolic Graph Retrieval** leverages hyperbolic geometry and deep learning to enable efficient and accurate retrieval in large-scale graph databases. By combining hyperbolic embeddings, transformer-inspired architectures, and generative adversarial hashing, this project achieves state-of-the-art performance in both speed and precision.

Key features include:

- **Hyperbolic Geometry**: Models hierarchical and structured information with low distortion.
- **Transformer-based Encoders**: Learns node and graph representations using custom hyperbolic attention mechanisms.
- **Generative Adversarial Hashing**: Produces compact binary codes for ultra-fast retrieval.
- **Scalable Hyperbolic K-Means Clustering**: Designed an efficient clustering algorithm in hyperbolic space, incorporating a custom cluster loss that minimizes the distance between each data point's hash and its corresponding cluster centroid hash—promoting tighter, more meaningful hash formations.
- **Flexible & Modular**: Easily extendable to new datasets and tasks.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Notebooks & Examples](#notebooks--examples)
- [Citation](#citation)
- [License](#license)

---

## Features

- **Hyperbolic Embedding**: Uses Lorentz model with [Geoopt](https://geoopt.readthedocs.io/) for efficient non-Euclidean representation.
- **HypFormer**: Custom transformer module with hyperbolic attention layers for graph encoding.
- **GAN-style Hashing**: Adversarial training for robust binary hash code generation in hyperbolic space.
- **Scalable Hyperbolic K-Means Clustering**: Designed an efficient clustering algorithm in hyperbolic space, incorporating a custom cluster loss that minimizes the distance between each data point's hash and its corresponding cluster centroid hash—promoting tighter, more meaningful hash formations.
- **Fast Retrieval**: Combines Hamming distance filtering and hyperbolic reranking for scalable search.
- **Extensive PyTorch Implementation**: Modular codebase for research and production.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nikhil-405/hyperbolic-graph-retrieval.git
   cd hyperbolic-graph-retrieval
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

   **Key Dependencies:**
   - `torch`
   - `geoopt`
   - `numpy`
   - `scipy`
   - `tqdm`
   - `jupyter` (for notebooks)

---

## Usage

### Training & Embedding

1. **Prepare Graph Embeddings**
   - Place your precomputed hyperbolic embeddings as a `.npy` file (e.g., `hyperbolic_embeddings.npy`) in the root directory.

2. **Run Main Training/Hashing Notebook**
   ```bash
   jupyter notebook "DS_GAN (1).ipynb"
   ```
   - This notebook demonstrates end-to-end training and retrieval.

3. **Retrieve Neighbors**
   - After training, use the provided retrieval functions to perform fast graph search with binary hash codes and hyperbolic (Lorentz) distance reranking.

### Example: Direct Python Usage

```python
import numpy as np
from DS_GAN import retrieve_neighbors

embeddings = np.load('hyperbolic_embeddings.npy')
hashes = np.load('hashes.npy')
query_idx = 7
retrieved = retrieve_neighbors(embeddings, hashes, query_idx)
print("Top retrieved indices:", retrieved)
```

---

## Code Structure

```
hyperbolic-graph-retrieval/
│
├── Hypformer/              # Core model code (HypFormer, attention, manifolds)
│   ├── attention.py
│   ├── hypformer.py
│   ├── main.py
│   └── manifolds/
│
├── embedding/              # Scripts for computing/saving embeddings (if any)
│
├── DS_GAN (1).ipynb        # End-to-end notebook: training, hashing, retrieval
├── requirements.txt
├── setup.py
└── .gitignore
```

- **Hypformer/**: Main model components and hyperbolic layers.
- **DS_GAN (1).ipynb**: Complete pipeline for training, hashing, clustering, and retrieval.
- **embedding/**: Utilities for graph embedding (optional/extendable).

---

## Notebooks & Examples

- **DS_GAN (1).ipynb**: Walks through hyperbolic embedding, GAN-based hashing, and retrieval.
- **Hypformer/main.py**: Example of using the HypFormer model standalone.

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{hyperbolicgraphretrieval2025,
  author    = {Nikhil and contributors},
  title     = {Scalable Hyperbolic Graph Retrieval with Deep Hashing},
  year      = {2025},
  url       = {https://github.com/nikhil-405/hyperbolic-graph-retrieval}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
