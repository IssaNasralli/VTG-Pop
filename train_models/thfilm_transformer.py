
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

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size=4, embed_dim=96):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        # x: (batch, H, W, C)
        x = self.proj(x)          # (batch, H/patch, W/patch, embed_dim)
        x = tf.reshape(x, [tf.shape(x)[0], -1, self.embed_dim])  # flatten patches
        x = self.norm(x)
        return x  # shape: (batch, num_patches, embed_dim)

class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super(AddPositionEmbedding, self).__init__()
        self.pos_emb = self.add_weight(
            "pos_emb",
            shape=[1, num_patches, embed_dim],
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )

    def call(self, x):
        return x + self.pos_emb

def transformer_block(x, embed_dim, num_heads=4, mlp_ratio=4, drop_rate=0.0):
    # LayerNorm + MultiHead Self Attention
    shortcut = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Add()([shortcut, x])

    # MLP
    shortcut = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(embed_dim * mlp_ratio, activation='gelu')(x)
    x = layers.Dense(embed_dim)(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Add()([shortcut, x])
    return x

def build_raster_transformer(raster_input,raster_input_size=96,patch_size=4,embed_dim=64,num_layers=2,num_heads=4,mlp_ratio=4,drop_rate=0.0):
    """
    Build the transformer encoder for raster input.

    Args:
        raster_input: Input tensor of shape (H, W, C)
        raster_input_size: Raster input spatial size (H = W)
        patch_size: Size of each patch
        embed_dim: Dimension of embedding space
        num_layers: Number of transformer encoder blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        drop_rate: Dropout rate

    Returns:
        Tensor: Final raster embedding vector of shape (batch, embed_dim)
    """
    # Patch embedding
    x = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)(raster_input)
    num_patches = (raster_input_size // patch_size) * (raster_input_size // patch_size)

    # Positional embedding
    x = AddPositionEmbedding(num_patches=num_patches, embed_dim=embed_dim)(x)

    # Transformer encoder blocks
    for _ in range(num_layers):
        x = transformer_block(x, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate)

    # Normalization and global pooling
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)

    # Linear projection to final embedding
    output = layers.Dense(embed_dim, activation='linear', name="raster_embedding")(x)

    return output

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
    
   # 🌍 Raster Transformer Encoder
    y = build_raster_transformer( raster_in,raster_input_size=raster_input_size, patch_size=patch_size,embed_dim=output_dim,num_layers=num_layers,num_heads=num_heads, mlp_ratio=mlp_ratio)

    
    
    # Final Feature extraction + Softplus for positive scalar output
    y = tf.expand_dims(y, axis=-1)
    p = layers.GlobalAveragePooling1D()(y) 
    p = layers.Dense(1, activation="softplus")(p)  # Positive scalar
    
    model = models.Model(
        inputs=[X_in, A_in, raster_in, node_index, flag_in],
        outputs=[p],
        name="ThFiLM-Net"
    )
    tf.print("========> Model Building (Only Transformer) Finish")

    return model

