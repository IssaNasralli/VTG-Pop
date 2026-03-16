# Node Feature Extraction

This module implements the **Node Feature Extraction** stage used in the VTG-Pop framework.  
Its objective is to compute **contextual attributes for each Functional Spatial Unit (FSU)** that will serve as node features in the spatial graph.

These features combine:

- **Intrinsic geometric properties** of the FSUs
- **Contextual indicators derived from raster layers**

The resulting feature table is used as input to the **Graph Neural Network (GNN)** component of VTG-Pop.

---

## Script

### `build_spatial_unit_features.py`

This script generates a **feature table for all spatial units** by combining information from the FSU shapefile and raster patches associated with each unit.

The output is a CSV file where each row corresponds to a **single spatial unit (graph node)** and each column represents a **node feature**.

---

## Processing Workflow

The script follows the pipeline below:

### 1. Load Functional Spatial Units

The program first loads the shapefile containing the Functional Spatial Units (FSUs):

unit_shp/unit_cleaned_filtered.shp


Each polygon corresponds to a spatial unit that will later become a **node in the spatial graph**.

---

### 2. Compute Geometry-Based Features

Several intrinsic attributes are computed directly from the polygon geometry:

- **Area** of the spatial unit (`area_m2`)
- **Unique node identifier** (`FID`)

These attributes capture structural characteristics of the spatial units.

---

### 3. Extract Core Attributes

A subset of relevant attributes is retained to build the feature table:

- `FID` : unique identifier of the spatial unit
- `ref_tn_cod` : administrative sector code
- `id_thiesse` : identifier of the Thiessen/Voronoi region
- `type` : urban/rural indicator
- `area_m2` : polygon area

The `type` attribute is converted to a numerical encoding:


urban -> 1
rural -> 0


---

### 4. Link Raster Patches to Spatial Units

Each spatial unit has an associated **multi-channel raster patch** stored in:


unit_patches_spyder/


The script automatically searches for the raster file corresponding to each `FID`.

---

### 5. Extract Raster-Derived Features

Two contextual indicators are extracted from the raster patches:

- **Road density mean**  
  Computed from **Band 7** of the raster patch.

- **School kernel density mean**  
  Computed from **Band 8** of the raster patch.

For each band, the script:

1. Reads the raster band
2. Masks no-data values
3. Computes the **mean value across the patch**

These statistics provide contextual information about **infrastructure and activity density** within each spatial unit.

---

### 6. Handle Missing Values

If a raster patch cannot be found or contains missing values, the corresponding features are set to:


0


This ensures a complete feature matrix for all spatial units.

---

### 7. Export Node Feature Table

The final feature table is saved as:


spatial_units_features.csv


Each row represents a spatial unit and includes:

| Feature | Description |
|------|------|
| FID | unique spatial unit identifier |
| ref_tn_cod | sector identifier |
| id_thiesse | Voronoi/Thiessen region id |
| type | urban/rural indicator |
| area_m2 | area of the spatial unit |
| road_density_mean | mean road density |
| school_kde_mean | mean school KDE |

This table is later used as the **node feature matrix** in the graph learning stage of VTG-Pop.

---

## Output Example


FID,ref_tn_cod,id_thiesse,type,area_m2,road_density_mean,school_kde_mean
```sql
0,1234,45,1,12450.3,0.52,0.18
1,1234,46,1,9876.5,0.47,0.21
...
```


---

## Role in the VTG-Pop Pipeline

This module provides the **contextual node features** required by the **Graph Neural Network** component of VTG-Pop.

These features complement the **visual embeddings extracted from raster patches** and enable the model to capture spatial relationships between neighboring units.

