# Data Collection

This folder contains the resources used to collect and organize the raw geospatial datasets required by the **VTG-Pop** framework. The data originate from multiple open sources including satellite imagery, census statistics, OpenStreetMap, and point-of-interest datasets.

The provided materials include Google Earth Engine scripts, tabular datasets, and download links for external geospatial layers.

---

## 1. Google Earth Engine Scripts

The following JavaScript files can be executed in the Google Earth Engine (GEE) Code Editor to access and export remote sensing layers used in the experiments.

- `DEM.js` : script for accessing and exporting the Digital Elevation Model (DEM).
- `NTL.js` : script for accessing and exporting Nighttime Lights (NTL) data.
- `MODIS.js` : script for accessing and exporting MODIS satellite imagery.
- `boundary_country.js` : script for retrieving the Tunisia country boundary from the GAUL dataset.

These scripts allow users to reproduce the raster layers used to construct the multi-channel spatial inputs of the VTG-Pop model.

---

## 2. School Location Dataset

- `school-GPS.csv`

This file contains the geographic coordinates of school locations used as points of interest (POIs) in the spatial analysis.

**Structure**

| Column | Description |
|------|-------------|
| `type` | Type of school |
| `lat` | Latitude |
| `long` | Longitude |

These locations are used to compute spatial features such as Kernel Density Estimation (KDE) layers representing functional activity within the study area.

---

## 3. Population Dataset

- `ins2024.xlsx`

This file contains the **Tunisia Census 2024 population data** obtained from the National Institute of Statistics (INS).

The file includes:

1. The **original census dataset**
2. A **custom cleaned sheet** prepared for this study

**Cleaned table structure**

| Column | Description |
|------|-------------|
| `code_gouvernorate` | Governorate code |
| `code_delegation` | Delegation code |
| `code_secteur` | Sector code |
| `secteur` | Sector name |
| `population` | Sector population |

This dataset provides the **sector-level population supervision used to train the VTG-Pop model**.

---

## 4. External Geospatial Data

Some datasets are too large to be stored directly in this repository. They can be downloaded from the following sources:

| Dataset | Description | Download Link |
|-------|-------------|--------------|
| OpenStreetMap (OSM) data | Road network and spatial features | LINK |
| DEM layer | Digital Elevation Model | LINK |
| NTL layer | Nighttime Lights imagery | LINK |
| MODIS imagery | Satellite-derived land surface information | LINK |
| Tunisia country boundary | GAUL country-level boundary | LINK |
| Tunisia sector boundary | Administrative sector boundaries | LINK |

Replace the placeholder `LINK` values with the appropriate download URLs.

---

## Notes

- All datasets were collected for research purposes in the context of the **VTG-Pop population estimation framework**.
- Some datasets may require preprocessing steps described in the `data_preprocessing/` module of this repository.
