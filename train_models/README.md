# 📂 train_models

This folder contains the training pipeline for all variants of the VTG-Pop models.

---

## ⚙️ 1. Requirements

Make sure your environment includes:

- Python ≥ 3.9  
- TensorFlow == 2.15  
- NumPy  
- Pandas  
- Rasterio  
- Keras  
- GeoPandas / GDAL  

### Install dependencies

    pip install tensorflow==2.15 numpy pandas rasterio keras geopandas

---

## ✅ 2. Pre-training Checklist

Before launching training, you **must execute the full data preparation pipeline**.

---

### Step 1 — Create Functional Spatial Units (FSU)

- Generate Voronoi-based FSUs
      This step was described in the FSU construction module:
    
[FSU_construction](https://github.com/IssaNasralli/VTG-Pop/tree/main/FSU_construction):

You can download the ready-to-use shape file (`unit.shp`) here: 👉 [FSUs](https://drive.google.com/file/d/1YHP2jMkBAQ1mZQe7XKywcjt17F43JzMU/view?usp=sharing)     
- Save the shapefile in:
    unit_shp/unit.shp

---

### Step 2 — Clean empty sector IDs

    python clean_empty_sector_id_from_unit_shp.py

---

### Step 3 — Filter INS and shapefile

    python filter_ins_and_shape.py

---

### Step 4 — Verify required files

Ensure the following files exist:

- ins_filtered.csv  
- sector.csv  

---

### Step 5 — Extract spatial unit data form the Multi-layer Raster Stack
```bash 
    python extract_spatial_unit_data.py
``` 

    
This step extracts raster-based features for each spatial unit using a **multi-layer raster stack**.
The raster stack used in this project is described in the data preprocessing module:
    
[data_preprocessing](https://github.com/IssaNasralli/VTG-Pop/tree/main/data_preprocessing):

You can download the ready-to-use raster file (`tunisia6.tif`) here: 

   👉 [Download](https://drive.google.com/file/d/18DKiQu5C6zlEOTOI4Hm4BccGmZHf0wnG/view?usp=sharing)
   
⚠️ Make sure to place the downloaded `.tif` file in the expected directory before running the script.

---

### Step 6 — Build spatial unit features

    python build_spatial_unit_features.py

---

## 🚀 3. Training Command

Run the training script with:

    python train.py x

Where:

- 1 → VTG-Pop (full model)  
- 2 → CNN baseline  
- 3 → Only GCN branch  
- 4 → Only Transformer branch  

Example:

    python train.py 1

---

## 🧠 4. Training Script Overview

The training is based on **sector-level weak supervision**.

---

### 🔹 Model Selection

The script loads different architectures depending on input:

- 1 → Full VTG-Pop (Transformer + GCN)  
- 2 → CNN baseline  
- 3 → GCN-only  
- 4 → Transformer-only  

---

### 🔹 Data Loading

The script loads:

- Graph adjacency matrix (A)  
- Node features (X)  
- Raster patches per spatial unit  
- Sector-level population labels  

---

### 🔹 Training Strategy

- Each sector contains multiple spatial units (FSUs)  
- The model predicts population per unit  
- Predictions are summed per sector  
- Loss is computed in log-space:

    loss = (sum(predictions) - log(1 + true_population))²

---

### 🔹 Data Augmentation

Raster inputs are randomly:

- Horizontally flipped  
- Vertically flipped  
- Rotated (0°, 90°, 180°, 270°)  

---

### 🔹 Dataset Split

Forced splits:
- Train → Great Tunis (prefixes 11–14)  
- Validation → Sousse (31)  
- Test → Sfax (34)  

Remaining data:
- 70% train  
- 15% validation  
- 15% test  

---

### 🔹 Hyperparameter Search

Grid search over:

- Learning rate  
- Dropout rate  
- L2 regularization  
- Hidden dimension  
- Output dimension  
- Number of heads  
- Number of layers  

Previously tested combinations are skipped.

---

### 🔹 Training Loop

For each configuration:

1. Iterate over sectors  
2. Compute sector-level loss  
3. Apply gradients  
4. Track mean training loss  

---

### 🔹 Validation & Early Stopping

- Validation loss computed each epoch  
- Best model is saved  
- Early stopping after 5 epochs without improvement  

---

### 🔹 Checkpoints

Saved in:

    checkpoints/

There's a checkpoint, for each model, provided to reuse for training.
---

### 🔹 Final Evaluation

- Best model is evaluated on test data  
- Outputs:
  - Log-MSE  
  - Log-RMSE  

---

### 🔹 Results Logging

All results are saved in:

    training_result.txt

---

## 📌 Notes

- Training is **sector-based**, not batch-based  
- Uses **weak supervision (census-level only)**  
- Supports **multi-modal geospatial data (graph + raster)**  

---
