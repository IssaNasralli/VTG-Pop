
import tensorflow as tf
from tensorflow.keras import Input, models
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Attention, MultiHeadAttention, Multiply, Add, Reshape, Lambda
import numpy as np
import os
import pickle
import pandas as pd
import networkx as nx
from spektral.layers import GATConv
from spektral.layers import GCNConv
from spektral.utils import normalized_adjacency, gcn_filter
import scipy.sparse as sp
import geopandas as gpd
from tensorflow.keras.regularizers import l2
import rasterio
def build_or_load_graph(shapefile_path, cache_prefix="graph_sparse"):
    """
    Build or load sparse adjacency from shapefile.
    Returns: nodes, edges, sparse adjacency (scipy.sparse.csr_matrix)
    """
    nodes_file = f"{cache_prefix}_nodes.pkl"
    edges_file = f"{cache_prefix}_edges.pkl"
    adj_file   = f"{cache_prefix}_adj_sparse.npz"

    # Load if cached
    if os.path.exists(nodes_file) and os.path.exists(edges_file) and os.path.exists(adj_file):
        print("📂 Found cached sparse graph → loading...")
        with open(nodes_file, "rb") as f:
            nodes = pickle.load(f)
        with open(edges_file, "rb") as f:
            edges = pickle.load(f)
        adj_sparse = sp.load_npz(adj_file)
        print(f"✅ Loaded {len(nodes)} nodes, {len(edges)} edges")
        return nodes, edges, adj_sparse

    # Else build from shapefile
    print("🔹 Building adjacency graph from shapefile...")
    gdf = gpd.read_file(shapefile_path)
    print(f"✅ Loaded {len(gdf)} polygons")
    gdf["FID"] = gdf.index

    G = nx.Graph()
    for idx in gdf.index:
        G.add_node(idx)

    geoms = list(gdf.geometry.items())
    N = len(geoms)
    for i, (idx1, geom1) in enumerate(geoms):
        for j in range(i + 1, N):
            idx2, geom2 = geoms[j]
            if geom1.touches(geom2):
                G.add_edge(idx1, idx2)
        if i % 100 == 0:
            print(f"• Processed {i}/{N} geometries")

    nodes = list(G.nodes)
    edges = list(G.edges)

    print(f"✅ Built graph with {len(nodes)} nodes and {len(edges)} edges")

    # Convert to sparse adjacency matrix
    adj_sparse = nx.to_scipy_sparse_array(G, nodelist=gdf.index, dtype=np.float32, format='csr')

    # Save for future runs
    with open(nodes_file, "wb") as f:
        pickle.dump(nodes, f)
    with open(edges_file, "wb") as f:
        pickle.dump(edges, f)
    sp.save_npz(adj_file, adj_sparse)

    print(f"✅ Saved sparse adjacency: {adj_sparse.shape}, {adj_sparse.nnz} nonzero elements")
    return nodes, edges, adj_sparse

def prepare_graph_data(shapefile_path="unit_shp/unit_cleaned_filtred.shp",feature_csv_path="spatial_units_features.csv",cache_prefix="graph_sparse"):
    """
    Load and prepare the graph data: adjacency (A), features (X), and feature_dim.

    Args:
        shapefile_path (str): Path to the shapefile for building/loading the graph.
        feature_csv_path (str): Path to CSV file containing node/tabular features.
        cache_prefix (str): Prefix for cached graph files.

    Returns:
        tuple: (A, X, feature_dim)
            A : tf.sparse.SparseTensor — normalized adjacency matrix
            X : tf.Tensor — node features tensor
            feature_dim : int — number of features per node
    """
    # 1️⃣ Build or load graph
    nodes, edges, adj_sparse = build_or_load_graph(shapefile_path, cache_prefix=cache_prefix)
    print(f"✅ Sparse adjacency shape: {adj_sparse.shape}, density = {adj_sparse.nnz / (adj_sparse.shape[0] ** 2):.6f}")

    # 2️⃣ Normalize adjacency (GNN stability)
    adj_norm = gcn_filter(adj_sparse)
    print("✅ Normalized adjacency computed")

    # 3️⃣ Load tabular data
    print("🔹 Loading tabular data...")
    tabular_df = pd.read_csv(feature_csv_path)
    print(f"✅ Loaded {len(tabular_df)} rows")

    # 4️⃣ Extract feature matrix
    feature_cols = ['area_m2', 'type',
                    'road_density_mean', 'school_kde_mean']
    features = tabular_df[feature_cols].values.astype(np.float32)
    num_nodes = features.shape[0]
    feature_dim = features.shape[1]
    print(f"✅ Features: {feature_dim} per node × {num_nodes} nodes")

    # 5️⃣ Convert to TensorFlow tensors
    X = tf.convert_to_tensor(features)

    if adj_norm.nnz < 5e6:
        A = tf.sparse.from_dense(tf.convert_to_tensor(adj_norm.todense()))
    else:
        indices = np.vstack(adj_norm.nonzero()).T
        A = tf.sparse.SparseTensor(
            indices=indices,
            values=adj_norm.data,
            dense_shape=adj_norm.shape
        )

    print("✅ Graph tensors prepared (A, X)")
    return A, X, feature_dim, num_nodes

def build_gcn_encoder(X_in, A_in, hidden_dim=32, output_dim=64, activation_hidden="elu", activation_output=None, dropout_rate=0.0, name_prefix="gcn"):
    """
    Build a basic GCN (Graph Convolution Network) encoder.

    Args:
        X_in (tf.Tensor): Node feature input tensor, shape (num_nodes, feature_dim)
        A_in (tf.SparseTensor): Sparse adjacency matrix tensor
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output embedding dimension
        activation_hidden (str): Activation for the hidden GCN layer
        activation_output (str or None): Activation for the output GCN layer
        dropout_rate (float): Dropout rate applied to GCN layers
        name_prefix (str): Prefix for naming the layers

    Returns:
        tf.Tensor: Output node embeddings of shape (num_nodes, output_dim)
    """
    print("🔹 Building GCN encoder...")

    # Hidden GCN layer
    x = GCNConv(
        hidden_dim,
        activation=activation_hidden,
        dropout_rate=dropout_rate,
        name=f"{name_prefix}_gcn_hidden"
    )([X_in, A_in])

    # Output GCN layer (embedding)
    x = GCNConv(
        output_dim,
        activation=activation_output,
        dropout_rate=dropout_rate,
        name=f"{name_prefix}_gcn_output"
    )([x, A_in])

    # Layer normalization
    x = layers.LayerNormalization(name=f"{name_prefix}_norm")(x)

    print("✅ GCN encoder built successfully")
    return x

def build_thfilm_inputs(num_nodes, feature_dim,raster_input_size=96,raster_input_number_bands=8):
    """
    Create and return the 5 Keras Input tensors used in the THFilm model.

    Args:
        feature_dim (int): Dimension of node feature vector.
        raster_input_size (int): Spatial size (height/width) of raster input.
        raster_input_number_bands (int): Number of raster channels/bands.

    Returns:
        tuple: (X_in, A_in, raster_in, node_index, flag_in)
    """
    X_in = Input(shape=(num_nodes, feature_dim), name="node_features")
    A_in = Input(shape=(None,), sparse=True, name="adjacency")
    raster_in = Input(
        shape=(raster_input_size, raster_input_size, raster_input_number_bands),
        name="raster_input"
    )
    #node_index = Input(shape=(), dtype=tf.int32, name="node_index")
    #flag_in = Input(shape=(), dtype=tf.int32, name="compute_flag")
    node_index = Input(shape=(1,), dtype=tf.int32, name="node_index")
    flag_in = Input(shape=(1,), dtype=tf.int32, name="compute_flag")
    return X_in, A_in, raster_in, node_index, flag_in

def build_thfilm(num_nodes, feature_dim, dropout_rate, l2_reg, hidden_dim, output_dim, num_heads, num_layers, raster_input_size=96, raster_input_number_bands=8, patch_size=4, mlp_ratio=4):
    # 🌟 Get all model inputs

    tf.print("========> Model Building Start")
    X_in, A_in, raster_in, node_index, flag_in = build_thfilm_inputs(num_nodes, feature_dim, raster_input_size, raster_input_number_bands)

    # ⚡ GNN Encoder
    z = build_gcn_encoder(X_in, A_in, hidden_dim=hidden_dim, output_dim=output_dim, activation_hidden="elu", activation_output=None, dropout_rate=dropout_rate, name_prefix="gcn")

    def conditional_gcn(inputs):
        z, node_idx, flag = inputs
        # Gather node embeddings
        z_sel = tf.gather(z, tf.cast(tf.squeeze(node_idx, axis=-1), tf.int32), axis=1)
        z_sel = tf.squeeze(z_sel, axis=1)
        # Use flag to select: if flag=1 -> keep z_sel, else -> keep previous value (here assume previous = z_sel)
        return tf.where(tf.equal(flag, 1), z_sel, z_sel)

    z_selected = Lambda(conditional_gcn)([z, node_index, flag_in])
    
    
    # Final Feature extraction + Softplus for positive scalar output
    z_selected = tf.expand_dims(z_selected, axis=-1)
    p = layers.GlobalAveragePooling1D()(z_selected)  
    p = layers.Dense(1, activation="softplus")(p)  # Positive scalar
    
    model = models.Model(
        inputs=[X_in, A_in, raster_in, node_index, flag_in],
        outputs=[p],
        name="ThFiLM-Net"
    )
    tf.print("========> Model Building (Only GCN) Finish")

    return model

