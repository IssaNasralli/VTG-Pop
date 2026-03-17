import os
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from thfilm import prepare_graph_data
from data import load_training_data
import json 
from keras.models import model_from_json
from data import load_training_data, split_sectors
import time
from datetime import timedelta
import itertools
import sys


model_choice = sys.argv[1]
prefix_model=""
if model_choice == "1":
    import thfilm as th
    print("✅ Using full ThFiLM model")
    prefix_model="full"

elif model_choice == "2":
    import thfilm_cnn as th
    print("✅ Using CNN baseline model")
    prefix_model="cnn"

elif model_choice == "3":
    import thfilm_gcn as th
    print("✅ Using only GCN Branch")
    prefix_model="gcn"
    
elif model_choice == "4":
    import thfilm_transformer as th
    print("✅ Using only Transformer Branch")
    prefix_model="gcn"

else:
    raise ValueError("Invalid option. Use 1 for full model, 2 for CNN baseline.")


def read_training_results(filename):
    results = []
    with open(filename, 'r') as file:
        # Skip the header line
        next(file)
        # Iterate over each line in the file
        for line in file:
            # Split the line by comma and extract the first three values
            learning_rate, dropout_rate, l2_reg,  hidden_dim,  output_dim,  num_heads,  num_layers, *_ = line.split(',')
            results.append((float(learning_rate), float(dropout_rate), float(l2_reg), float(hidden_dim), float(output_dim), float(num_heads), float(num_layers)))
    return results 

def augment_raster(raster):
    """
    Apply random data augmentation:
    - horizontal flip
    - vertical flip
    - rotation (0, 90, 180, 270)
    """
    raster = tf.convert_to_tensor(raster)

    # Random horizontal flip
    raster = tf.image.random_flip_left_right(raster)

    # Random vertical flip
    raster = tf.image.random_flip_up_down(raster)

    # Random rotation (k * 90°)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    raster = tf.image.rot90(raster, k)

    return raster
    
def compute_sector_loss(model, sector_code, sector_to_fids, sector_rasters, X, A, sector_population):
    """
    Forward pass for one sector.
    Predicts each spatial unit, sums predictions, compares with true (log) population.
    'sector_rasters' contains only the rasters of this sector (FID→array).
    """
    unit_ids = sector_to_fids[sector_code]
    true_pop = float(sector_population[sector_code])
    true_log_pop = tf.math.log1p(true_pop)

    X_full = tf.expand_dims(X, axis=0)
    A_full = tf.sparse.expand_dims(A, axis=0)

    preds = []
    flag_arr = np.array([1], dtype=np.int32)
    first = True

    for fid in unit_ids:
        if fid not in sector_rasters:
            continue  # skip missing rasters (shouldn’t happen)
            
        raster = np.expand_dims(sector_rasters[fid], axis=0)
        raster = augment_raster(raster)
        node_idx = np.array([fid], dtype=np.int32)

        p = model([
            X_full,
            A_full,
            raster,
            node_idx,
            flag_arr
        ], training=True)
        preds.append(p[0][0])

        if first:
            flag_arr = np.array([0], dtype=np.int32)
            first = False

    if not preds:
        return tf.constant(0.0, dtype=tf.float32)

    preds = tf.stack(preds)
    pred_sector_pop = tf.reduce_sum(preds)

    loss = tf.square(pred_sector_pop - true_log_pop)
    return loss

def evaluate_sectors(model, sectors, sector_to_fids, rasters, X, A, sector_population):
    """
    Compute mean log-space MSE loss across given sectors.
    Used for validation or testing.
    """
    X_full = tf.expand_dims(X, axis=0)
    A_full = tf.sparse.expand_dims(A, axis=0)
    losses = []
    n_sectors = len(sectors)
    i=0   
    for sec in sectors:
        i=i+1

        print(f"\r Evaluating, sector {i:04d}/{n_sectors} ...", end="", flush=True)
        
        if sec not in sector_to_fids or sec not in sector_population:
            continue
        unit_ids = sector_to_fids[sec]
        true_log_pop = tf.math.log1p(float(sector_population[sec]))

        preds = []
        flag_arr = np.array([1], dtype=np.int32)
        first = True

        for fid in unit_ids:
            raster = np.expand_dims(rasters[fid], axis=0)
            node_idx = np.array([fid], dtype=np.int32)
            p = model([
                X_full,
                A_full,
                raster,
                node_idx,
                flag_arr
            ], training=False)
            preds.append(p[0][0])

            if first:
                flag_arr = np.array([0], dtype=np.int32)
                first = False

        pred_sector_pop = tf.reduce_sum(tf.stack(preds))
        losses.append(tf.square(pred_sector_pop - true_log_pop).numpy())

    return float(np.mean(losses)) if losses else np.nan


if __name__ == "__main__":

    random_seed = 42
    EPOCH = 6

    RASTER_FOLDER = "unit_patches_spyder"
    RASTER_INPUT_SIZE = 96
    FEATURE_CSV = "spatial_units_features.csv"
    INS_CSV = "ins_filtered.csv"
    SECTOR_CSV = "sector.csv"
    shapefile_path = "unit_shp/unit_cleaned_filtered.shp"

    A, X, rasters, sector_population, sector_to_fids, all_fids, feature_dim, num_nodes = load_training_data(
        shapefile_path=shapefile_path,
        raster_folder=RASTER_FOLDER,
        feature_csv=FEATURE_CSV,
        ins_csv=INS_CSV,
        sector_csv=SECTOR_CSV,
        raster_input_size=RASTER_INPUT_SIZE,
        cache_prefix="graph_sparse"
    )

    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    # ---------------------------------------------------------
    # Prefix definitions
    # ---------------------------------------------------------

    TRAIN_PREFIXES = ["11", "12", "13", "14"]   # Great Tunis
    VAL_PREFIX = "31"                           # Sousse
    TEST_PREFIX = "34"                          # Sfax

    # ---------------------------------------------------------
    # Extract forced TRAIN sectors
    # ---------------------------------------------------------

    train_population_prefix = {
        sec: pop for sec, pop in sector_population.items()
        if any(str(sec).startswith(p) for p in TRAIN_PREFIXES)
    }

    train_fids_prefix = {
        sec: fids for sec, fids in sector_to_fids.items()
        if any(str(sec).startswith(p) for p in TRAIN_PREFIXES)
    }

    # ---------------------------------------------------------
    # Extract forced VALIDATION sectors
    # ---------------------------------------------------------

    val_population_prefix = {
        sec: pop for sec, pop in sector_population.items()
        if str(sec).startswith(VAL_PREFIX)
    }

    val_fids_prefix = {
        sec: fids for sec, fids in sector_to_fids.items()
        if str(sec).startswith(VAL_PREFIX)
    }

    # ---------------------------------------------------------
    # Extract forced TEST sectors
    # ---------------------------------------------------------

    test_population_prefix = {
        sec: pop for sec, pop in sector_population.items()
        if str(sec).startswith(TEST_PREFIX)
    }

    test_fids_prefix = {
        sec: fids for sec, fids in sector_to_fids.items()
        if str(sec).startswith(TEST_PREFIX)
    }

    # ---------------------------------------------------------
    # Remove forced sectors from original dictionaries
    # ---------------------------------------------------------
    forced_sectors = (
        set(train_population_prefix.keys()) |
        set(val_population_prefix.keys()) |
        set(test_population_prefix.keys())
    )
    saved_sector_population=sector_population
    sector_population = {
        sec: pop for sec, pop in sector_population.items()
        if sec not in forced_sectors
    }



    print("Forced TRAIN sectors (Great Tunis):", len(train_population_prefix))
    print("Forced VAL sectors (Sousse):", len(val_population_prefix))
    print("Forced TEST sectors (Sfax):", len(test_population_prefix))
    print("Remaining sectors for random split:", len(sector_population))

    # ---------------------------------------------------------
    # Random split on remaining sectors
    # ---------------------------------------------------------

    train_sectors, val_sectors, test_sectors = split_sectors(
        sector_population,
        sector_to_fids,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=random_seed
    )
    sector_population=saved_sector_population

    # ---------------------------------------------------------
    # Add forced sectors to their respective splits
    # ---------------------------------------------------------

    train_sectors = list(train_sectors) + list(train_population_prefix.keys())
    val_sectors = list(val_sectors) + list(val_population_prefix.keys())
    test_sectors = list(test_sectors) + list(test_population_prefix.keys())

    print("Final splits:")
    print("Train sectors:", len(train_sectors))
    print("Validation sectors:", len(val_sectors))
    print("Test sectors:", len(test_sectors))
    
    # ============================================================
    #  Training loop (sector-level supervision)
    # ============================================================
    print("\n Starting training loop...\n")
    
    
    learning_rate_values = [1e-3, 5e-4, 1e-4]
    dropout_rate_values = [0.1, 0.2]
    l2_reg_values = [1e-5, 1e-4]
    hidden_dim_values = [32, 64]
    output_dim_values = [64, 128]
    num_heads_values = [2, 4]
    num_layers_values = [1, 2]

    training_results_file="training_result.txt"
    iterated_values = read_training_results(training_results_file)

    print("Already tested:", iterated_values)

    with open(training_results_file, 'a') as f:
        for (learning_rate, dropout_rate, l2_reg,
             hidden_dim, output_dim, num_heads, num_layers) in itertools.product(
                learning_rate_values, dropout_rate_values, l2_reg_values,
                hidden_dim_values, output_dim_values, num_heads_values, num_layers_values):

            combo = (learning_rate, dropout_rate, l2_reg,
                     hidden_dim, output_dim, num_heads, num_layers)

            if combo in iterated_values:
                print(f"Skipping: lr={learning_rate}, drop={dropout_rate}, l2={l2_reg}, "
                      f"h_dim={hidden_dim}, out_dim={output_dim}, heads={num_heads}, layers={num_layers}")
                continue
            else:
                print(f"Running: lr={learning_rate}, drop={dropout_rate}, l2={l2_reg}, "
                      f"h_dim={hidden_dim}, out_dim={output_dim}, heads={num_heads}, layers={num_layers}")

            model = th.build_thfilm(num_nodes,feature_dim, dropout_rate=dropout_rate, l2_reg=l2_reg, hidden_dim=hidden_dim, output_dim=output_dim, num_heads=num_heads, num_layers=num_layers)
            model.compile(run_eagerly=False) 
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
            checkpoint_path = f"checkpoints/{prefix_model}model_lr{learning_rate}_drop{dropout_rate}_l2{l2_reg}_hidden_dim{hidden_dim}_output_dim{output_dim}_heads{num_heads}_layers{num_layers}.h5"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            # === Load weights if available ===
            if os.path.exists(checkpoint_path):
                model.load_weights(checkpoint_path)
                print(f"✅ Found existing weights — loaded from: {checkpoint_path}")
            else:
                print(f" No weights found at {checkpoint_path} — training from scratch.")

            best_val_loss = float("inf")
            patience = 5          # epochs without improvement before stopping
            wait = 0
    

            for epoch in range(EPOCH):
                # Record full epoch start time
                epoch_start = time.time()
                epoch_losses = []

                np.random.shuffle(train_sectors)

                # ============================================================
                # ️ Training pass
                # ============================================================
                t_train_start = time.time()
                n_train = len(train_sectors)
                for i, sec in enumerate(train_sectors, 1):


                    sector_rasters = {
                                        fid: rasters.get(fid)
                                        for fid in sector_to_fids.get(sec, [])
                                        if fid in rasters
                                    }
                    with tf.GradientTape() as tape:
                        loss = compute_sector_loss(model, sec, sector_to_fids, sector_rasters, X, A, sector_population)
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    epoch_losses.append(loss.numpy())
                    mean_train = np.mean(epoch_losses)
                    print(f"\r Epoch {epoch+1}  Training sector {i:04d}/{n_train} ...Mean epoch losses:{mean_train:.4f}", end="", flush=True)


                mean_train = np.mean(epoch_losses)
                train_time = time.time() - t_train_start
                print(f"\r✅ Training pass complete ({n_train} sectors, {timedelta(seconds=int(train_time))})")

                # ============================================================
                #  Validation pass
                # ============================================================
                t_val_start = time.time()
                rasters_val = {
                    fid: rasters[fid]
                    for sec in val_sectors
                    for fid in sector_to_fids.get(sec, [])
                    if fid in rasters
                }
                mean_val = evaluate_sectors(model, val_sectors, sector_to_fids, rasters_val, X, A, sector_population)
                val_time = time.time() - t_val_start

                print(f" Epoch {epoch+1:02d}/{EPOCH} — "
                      f"train_loss: {mean_train:.4f} — val_loss: {mean_val:.4f} "
                      f"(train {timedelta(seconds=int(train_time))}, val {timedelta(seconds=int(val_time))})")

                # ============================================================
                #  Checkpoint & early stopping
                # ============================================================
                if mean_val < best_val_loss:
                    best_val_loss = mean_val
                    wait = 0
                    model.save_weights(checkpoint_path)
                    print(f" New best model saved (val_loss={best_val_loss:.4f})")
                else:
                    wait += 1
                    if wait >= patience:
                        print(f" Early stopping triggered (no improvement for {patience} epochs).")
                        break

                epoch_time = time.time() - epoch_start
                print(f" Epoch {epoch+1:02d} completed in {timedelta(seconds=int(epoch_time))}\n")

            # ============================================================
            #  Evaluate best model on test sectors
            # ============================================================
            if os.path.exists(checkpoint_path):
                model.load_weights(checkpoint_path)
                print(f"✅ Loaded best model weights from {checkpoint_path}")

            t_test_start = time.time()
            rasters_test = {
                fid: rasters[fid]
                for sec in test_sectors
                for fid in sector_to_fids.get(sec, [])
                if fid in rasters
            }

            test_loss = evaluate_sectors(model, test_sectors, sector_to_fids, rasters_test, X, A, sector_population)
            test_time = time.time() - t_test_start

            print(f"\n Final test loss (log-space MSE): {test_loss:.4f}")
            print(f" Approx. log-RMSE: {np.sqrt(test_loss):.4f}")
            print(f" Test evaluation time: {timedelta(seconds=int(test_time))}")
            
            # Save the results to the text file
            f.write(f"\n{learning_rate}, {dropout_rate}, {l2_reg}, {hidden_dim}, {output_dim}, {num_heads}, {num_layers}, {mean_train}, {mean_val}, {test_loss}")
            #f.write(f"{learning_rate}, {learning_rate}, {l2_reg}, {hidden_dim}, {output_dim}, {num_heads}, {num_layers}, {mean_train}, {mean_val}, {test_loss}\n")             
            f.flush()  # Ensure the data is written to the file
            os.fsync(f.fileno())
