from thfilm import prepare_graph_data
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
import random


def load_training_data(shapefile_path, raster_folder, feature_csv, ins_csv, sector_csv, raster_input_size,cache_prefix="graph_sparse"):
    """
    Returns:
      A: tf.sparse.SparseTensor or tf.SparseTensor-like adjacency (normalized)
      X: tf.Tensor of shape (num_nodes, feature_dim)
      rasters: dict {FID: np.array((H,W,C), float32)} resized to raster_input_size
      sector_population: dict {code_sec: population}
      sector_to_fids: dict {code_sec: [fid1, fid2, ...]}
      all_fids: np.array of FIDs (order corresponding to X rows where possible)
    """
    print("🔹 Preparing graph (A, X) using thfilm.prepare_graph_data() ...")
    A, X, feature_dim, num_nodes = prepare_graph_data(
        shapefile_path=shapefile_path,
        feature_csv_path=feature_csv,
        cache_prefix=cache_prefix
    )
    print(f"✅ Graph ready — num_nodes={num_nodes}, feature_dim={feature_dim}")
    print("🔹 Loading tabular CSV to map FID → sector and assemble FID list ...")
    df = pd.read_csv(feature_csv)
    # Expect columns like: FID, ref_tn_cod, id_thiesse, type, area_m2, ...
    if "FID" not in df.columns:
        raise ValueError("spatial_units_features.csv must contain 'FID' column")
    if "ref_tn_cod" not in df.columns:
        raise ValueError("spatial_units_features.csv must contain 'ref_tn_cod' column (sector code)")

    all_fids = df["FID"].astype(int).values
    ref_codes = df["ref_tn_cod"].astype(int).values

    # Build sector -> list of FIDs
    sector_to_fids = {}
    for fid, sec in zip(all_fids, ref_codes):
        sector_to_fids.setdefault(sec, []).append(int(fid))

    print(f"✅ Found {len(sector_to_fids)} sectors and {len(all_fids)} spatial units")
    # ✅ Found 2067 sectors and 25508 spatial units
    # Load sector populations (ins.csv)
    if not os.path.exists(ins_csv):
        raise FileNotFoundError(f"Missing sector population file: {ins_csv}")
    ins_df = pd.read_csv(ins_csv)
    if "code_sec" not in ins_df.columns or "population" not in ins_df.columns:
        raise ValueError("ins.csv must contain 'code_sec' and 'population' columns")
    sector_population = dict(zip(ins_df["code_sec"].astype(int).values,
                                 ins_df["population"].astype(float).values))
    print(f"✅ Loaded population for {len(sector_population)} sectors from {ins_csv}")

    # read sector.csv 
    if os.path.exists(sector_csv):
        sector_df = pd.read_csv(sector_csv)
        print(f"🔹 Loaded {len(sector_df)} rows from {sector_csv}")
    else:
        sector_df = None

    # Load rasters for every FID found in features CSV
    print(f"🔹 Loading raster patches from {raster_folder}, this may use ~1 GiB RAM...")
    rasters = {}
    missing = []
    
    # Expect raster filenames like "FID_N.tif"
    for fid in all_fids:
        raster_path = os.path.join(raster_folder, f"FID_{fid}.tif")
        if not os.path.exists(raster_path):
            missing.append(raster_path)
            continue
        with rasterio.open(raster_path) as src:
            arr = src.read()                      # (bands, H, W)
            arr = np.transpose(arr, (1, 2, 0))    # -> (H, W, C)
            # Resize spatially to model input size (use TF to avoid extra deps)
            arr_resized = tf.image.resize(arr, (raster_input_size, raster_input_size), method='bilinear').numpy()
            # Normalize or cast to float32 (you can adjust normalization later)
            rasters[int(fid)] = arr_resized.astype(np.float32)

    if missing:
        print(f"⚠️ Warning: {len(missing)} missing raster files (examples): {missing[:3]}")
    
    print(f"✅ Loaded rasters for {len(rasters)} / {len(all_fids)} FIDs")

    return A, X, rasters, sector_population, sector_to_fids, all_fids,feature_dim,num_nodes

def split_sectors(sector_population, sector_to_fids, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split the available sectors into train / validation / test subsets.

    Args:
        sector_population : dict {code_sec: population}
        sector_to_fids    : dict {code_sec: [fid1, fid2, ...]}
        train_ratio       : float (default 0.7)
        val_ratio         : float (default 0.15)
        test_ratio        : float (default 0.15)
        seed              : int, random seed for reproducibility

    Returns:
        train_sectors, val_sectors, test_sectors : lists of sector codes
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    sectors = list(sector_population.keys())
    random.Random(seed).shuffle(sectors)
    n_total = len(sectors)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    train_sectors = sectors[:n_train]
    val_sectors   = sectors[n_train:n_train + n_val]
    test_sectors  = sectors[n_train + n_val:]

    print(f"📊 Sector split → train={len(train_sectors)}  "
          f"val={len(val_sectors)}  test={len(test_sectors)}  (total={n_total})")

    # Sanity check: verify sector codes exist in mappings
    missing = [s for s in train_sectors + val_sectors + test_sectors
               if s not in sector_to_fids]
    if missing:
        print(f"⚠️ Warning: {len(missing)} sectors not found in sector_to_fids")

    # ✅ Export missing sectors to file
    with open("missing_sectors.txt", "w") as f:
        for m in missing:
            f.write(str(m) + "\n")
    print("📝 Missing sectors exported to missing_sectors.txt")

    # ✅ Export the sector_to_fids mapping
    with open("sector_to_fids.txt", "w") as f:
        for sec, fids in sector_to_fids.items():
            f.write(f"{sec}: {','.join(map(str, fids))}\n")
    print("📝 sector_to_fids mapping exported to sector_to_fids.txt")

    return train_sectors, val_sectors, test_sectors
