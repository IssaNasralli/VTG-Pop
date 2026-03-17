
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

def build_thfilm(num_nodes, feature_dim, dropout_rate, l2_reg, hidden_dim, output_dim, num_heads, num_layers, 
                 raster_input_size=96, raster_input_number_bands=8, patch_size=4, mlp_ratio=4):
    # 🌟 Get all model inputs (keep interface unchanged)
    X_in, A_in, raster_in, node_index, flag_in = build_thfilm_inputs(num_nodes, feature_dim, raster_input_size, raster_input_number_bands)
    
    # 🔹 CNN branch on raster input
    x = raster_in
    x = layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    p = layers.Dense(1, activation="softplus")(x)  # Positive scalar prediction

    # Build model (inputs unchanged)
    model = models.Model(
        inputs=[X_in, A_in, raster_in, node_index, flag_in],
        outputs=[p],
        name="CNN-Baseline"
    )
    
    tf.print("========> CNN Baseline Model Built Successfully")
    return model