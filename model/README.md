# models/

This directory contains the implementation of the **VTG-Pop architecture**, the core model used for fine-grained population estimation.

The main file in this folder is:


thfilm.py


which implements the hybrid **Transformer–Graph Neural Network** model used in the paper.

---

# thfilm.py

`thfilm.py` implements the neural architecture used by **VTG-Pop**.  
The model integrates three complementary components:

1. **Graph Neural Network (GNN)** for spatial context modeling
2. **Transformer encoder** for raster feature extraction
3. **FiLM conditioning** to fuse graph embeddings with visual features

The final output is a **population estimate for a given Functional Spatial Unit (FSU)**.

---

# High-Level Pipeline

The model follows the pipeline below:

```sql

Raster Patch (96×96×8)
│
▼
Transformer Encoder
│
Raster Embedding
│
▼
FiLM Modulation ◄──── Node Embedding from GNN
│
▼
Population Regression Head
│
▼
Predicted Population
```

The **Graph Neural Network** processes the full spatial graph of FSUs, producing node embeddings that represent the spatial context of each unit.  
These embeddings condition the raster features through **Feature-wise Linear Modulation (FiLM)** before the final prediction.

---

# Main Components

## 1. Graph Construction

The graph is built from the FSU shapefile using spatial adjacency:

```sql
build_or_load_graph()
```

Nodes represent **Functional Spatial Units**, and edges connect **adjacent polygons**.  
The adjacency matrix is stored in **sparse format** and cached to avoid rebuilding the graph at every run.

---

## 2. Graph Data Preparation

```sql
prepare_graph_data()
```

This function prepares the graph inputs used by the model:

- normalized adjacency matrix
- node feature matrix
- feature dimensionality

Node features include spatial attributes such as:

- polygon area
- unit type
- road density
- school kernel density

---

## 3. Graph Neural Network Encoder

The spatial relationships between FSUs are modeled using a **Graph Convolution Network (GCN)**:


```sql

build_gcn_encoder()
```

The encoder produces a **node embedding** for each FSU that captures spatial dependencies among neighboring units.

An alternative **Graph Attention Network (GAT)** encoder is also provided:

```sql

build_gat_encoder()
```

---

## 4. Raster Transformer Encoder

Raster inputs are processed using a **Vision Transformer–style encoder**:


build_raster_transformer()


Steps:
```sql
1. Patch embedding of the raster image
2. Positional encoding
3. Multiple transformer blocks with self-attention
4. Global pooling to produce a compact raster embedding
```

The output is a **vector representation of the spatial patterns** contained in the raster patch.

---

## 5. FiLM Feature Modulation

Raster embeddings are conditioned using graph node embeddings via **Feature-wise Linear Modulation (FiLM)**:


film_modulation()


This module learns two vectors:

- **γ (gamma)** – feature scaling
- **β (beta)** – feature shifting

The modulated features are computed as:

```sql
y_mod = γ ⊙ y + β

```
where `y` is the raster embedding and `⊙` denotes element-wise multiplication.

---

## 6. Final Population Prediction

After FiLM modulation, the model applies:

- global pooling
- a fully connected layer
- a **Softplus activation**

to produce a **positive scalar population estimate**.

---

# Model Inputs

The model expects five inputs:

| Input | Description |
|-----|-----|
| `X_in` | Node feature matrix |
| `A_in` | Sparse adjacency matrix |
| `raster_in` | Raster patch for the target FSU |
| `node_index` | Index of the FSU node in the graph |
| `flag_in` | Control flag used during sector-wise batching |

---

# Model Output

The model outputs:


p = predicted population of the target FSU


During training, predictions from all FSUs belonging to the same sector are aggregated to match the available census supervision.

---

# Model Builder

The complete VTG-Pop model is constructed using:


build_thfilm()


This function assembles:

- the **raster Transformer encoder**
- the **graph neural network encoder**
- the **FiLM conditioning module**
- the **population regression head**

into a single TensorFlow/Keras model.

---

# Dependencies

The model relies on the following libraries:

- TensorFlow / Keras
- Spektral (graph neural networks)
- NetworkX
- GeoPandas
- Rasterio
- SciPy

---

# Notes

The architecture implemented in `thfilm.py` corresponds to the **VTG-Pop framework** described in the accompanying research paper.

The implementation is designed to efficiently handle large spatial graphs while processing raster inputs associated with individual FSUs.

If
