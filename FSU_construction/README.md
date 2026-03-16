# FSU Generation

This folder contains the workflow used to generate the **Functional Spatial Units (FSUs)** used by the VTG-Pop framework.

FSUs are constructed by combining three spatial layers:

- Administrative **sectors**
- **Voronoi (Thiessen) polygons** derived from Points of Interest (POIs)
- **Major road network**

The process consists of two main GIS steps performed in **ArcGIS**, followed by data cleaning using **Python scripts**.

---

# 1. Generate Voronoi (Thiessen) Polygons from POIs

The first step consists of generating **Voronoi polygons** from the POI layer. These polygons represent areas of influence around each POI.

### Steps in ArcGIS

1. Load the **POI shapefile** into ArcGIS.
2. Open **ArcToolbox**.
3. Navigate to:
   ```sql
 Analysis Tools → Proximity → Create Thiessen Polygons```
5. 
4. Configure the tool parameters:

| Parameter | Value |
|----------|------|
| Input Features | POI shapefile |
| Output Feature Class | `voronoi.shp` |
| Fields to Copy | `ALL` |

5. Run the tool.

This produces a **Voronoi tessellation** covering the entire study area.

---

# 2. Generate Functional Spatial Units (FSUs)

The final FSUs are created by intersecting:

- Administrative **sector boundaries**
- **Major roads**
- **Voronoi polygons**

This intersection produces smaller spatial units that respect:

- administrative limits
- functional activity zones
- urban morphology defined by road segmentation

### Steps in ArcGIS

1. Load the following layers:
sectors.shp
major_roads.shp
voronoi.shp
2. Split sectors using the **major roads layer**.
Use:

```Analysis Tools → Overlay → Identity 
```
or:
```
Analysis Tools → Overlay → Intersect
```

4. Intersect the resulting layer with the **Voronoi polygons**.

Tool:
```
Analysis Tools → Overlay → Intersect
```
Input layers:
sector_split_by_roads.shp
voronoi.shp
Output:
unit.shp


This layer contains the **initial Functional Spatial Units (FSUs)**.

Each polygon represents a spatial unit defined by:

- administrative sector
- Voronoi influence area
- road network segmentation

---

# 3. Cleaning FSUs with Missing Sector IDs

Some polygons in `unit.shp` may not contain a valid sector identifier (`ref_tn_cod`).  
These polygons must be removed before further processing.

This is handled by the script:

clean_empty_sector_id_from_unit_shp.py

### Script Description

The script performs the following steps:

1. Load the shapefile `unit.shp` using **GeoPandas**
2. Convert the `ref_tn_cod` field to string format for safe comparison
3. Remove polygons where the sector code is:
   - empty (`""`)
   - `"0"`
   - `"nan"`
   - `"None"`
   - or actual `NaN`
4. Save the cleaned shapefile

Output:
unit_cleaned.shp

This ensures that every remaining FSU polygon is associated with a valid administrative sector.

---

# 4. Filtering CSV and Shapefile Consistency

The script: filter_ins_and_shape.py

ensures consistency between:

- the **population dataset (`ins.csv`)**
- the **FSU shapefile (`unit_cleaned.shp`)**

### Script Logic

The script performs the following operations:

### Step 1 — Filter the population CSV

- Load `ins.csv`
- Keep only rows where `code_sec` exists in the shapefile attribute `ref_tn_cod`

Result:
ins_filtered.csv

### Step 2 — Filter the FSU shapefile

- Keep only polygons whose `ref_tn_cod` exists in the filtered CSV

Result:
unit_cleaned_filtered.shp

### Output Files
ins_filtered.csv
unit_shp/unit_cleaned_filtered.shp


This guarantees that:

- every sector appearing in the population data exists in the spatial dataset
- every FSU polygon belongs to a valid sector

---

# 5. Data Files

### FSU Shapefiles

The shapefiles used in the experiments are available here: 👉 [FSUs]([https://drive.google.com/file/d/1x6UsGGfNMw45f9-rS6nmHXYdcthIvmQ5/view?usp=sharing](https://drive.google.com/file/d/1YHP2jMkBAQ1mZQe7XKywcjt17F43JzMU/view?usp=sharing))  


The archive contains:
unit.shp → Original FSU layer
unit_cleaned.shp → Cleaned FSU layer
unit_cleaned_filtred.shp → Cleaned and Filtred FSU layer

### Population Dataset

The file ins_filtered.csv is provided directly in this repository. It contains the filtered sector-level population dataset used for training and evaluation.

---

# Output of this Folder

After completing the workflow, the following files are produced:
unit.shp
unit_cleaned.shp
unit_cleaned_filtered.shp
ins_filtered.csv


These files are used as the spatial foundation for building the **FSU graph structure** and training the **VTG-Pop population estimation model**.


   

