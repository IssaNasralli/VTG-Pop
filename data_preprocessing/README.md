# `data_preprocessing/`

This folder contains preprocessing pipelines for generating raster layers and spatial features used by the VTG-Pop model. It includes workflows for **POIs**, **road networks**, and **normalization scripts** for raster layers.

---

## 1. Preprocessing Points of Interest (POI)

The POI dataset is processed to create a **Kernel Density Estimation (KDE) raster** representing the spatial concentration of facilities (e.g., schools), which serves as an indicator of population distribution.

### Processing Steps (ArcGIS Pro)

1. Import the tabular dataset.
2. Convert the table to **point vector data**.
3. Apply **Kernel Density Estimation (KDE)**.

**Parameters used:**

- Search radius (bandwidth): 5 km  
- Method: Planar  
- Area units: Square kilometers  

**Download Links:**

- Shapefile: 👉 [POI Points Shapefile](https://drive.google.com/file/d/1x6UsGGfNMw45f9-rS6nmHXYdcthIvmQ5/view?usp=sharing)  
- Raster KDE: 👉 [POI KDE Raster](https://drive.google.com/file/d/15iIxcrfq27usYyEuME2DCR9-FEcY7P7a/view?usp=sharing)

---

## 2. Preprocessing Road Networks

Road network layers are preprocessed to capture **major roads** and generate a **road intersection density raster** used as a spatial feature.

### 2.1 Filtering Major Roads

Use the following SQL-like filter:

```sql
highway IN ('motorway', 'trunk', 'primary', 'secondary', 'tertiary','service')
```
### 2.2 Snapping Road Features
A Python script (snap.py) is provided to snap road vertices and endpoints to ensure topological consistency.
### 2.3 Road Intersection Density Raste
After snapping, compute the road intersection density raster. A ready-to-use raster is available here:
👉 [Road Density Raster](https://drive.google.com/file/d/1_MBve03Jgqh52FRAdLxEeUbEz1usdr4r/view)
## 3. Normalization Scripts

To prepare raster layers for the VTG-Pop model, normalization scripts are provided:

norm_single_band.py – Normalizes single-band raster layers (e.g., NTL, road intersection density, DEM, slope).

norm_modis.py – Normalizes three-band raster imagery (MODIS RGB layers).

These scripts scale raster values to a suitable range for model input.



## 4. Notes

All preprocessing steps follow the methodology described in the paper for generating multi-layer raster stacks and functional spatial feature.

Users can replace input datasets with their own geospatial data, but the preprocessing steps must be applied consistently to ensure compatibility with VTG-Pop.
