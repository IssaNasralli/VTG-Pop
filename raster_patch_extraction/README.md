
# Raster Patch Extraction

This module extracts raster patches corresponding to each Functional Spatial Unit (FSU) from the multi-layer raster stack used by the VTG-Pop framework.

The script `extract_spatial_unit_data.py` clips the input raster according to the geometry of each spatial unit polygon and exports an individual raster patch for every FSU.

## Purpose

In the VTG-Pop pipeline, each FSU is represented by a local raster context extracted from a multi-channel raster dataset. These raster patches are later used as inputs to the visual encoder (Transformer branch) of the model.

This module automates the extraction of these patches by iterating over all spatial unit polygons and masking the raster accordingly.

## Input Data

The script requires the following inputs:

- **Multi-layer raster stack**  
  A GeoTIFF raster file containing all raster channels used for the model.  
  Example:

tunisia6.tif


- **Spatial unit polygons**  
A shapefile containing the geometries of the Functional Spatial Units (FSUs).  
Example:

unit_shp/unit_cleaned_filtered.shp


Each polygon in the shapefile corresponds to a spatial unit for which a raster patch will be extracted.

## Output

The script generates one raster patch per spatial unit.

Output directory:


unit_patches_spyder/


Each patch is saved as a GeoTIFF file:
```sql

FID_0.tif
FID_1.tif
FID_2.tif
...
```


The filename index corresponds to the polygon index in the GeoDataFrame.

Each output raster:

- contains the same raster channels as the input stack
- is clipped to the extent of the spatial unit
- preserves the original geospatial metadata (transform and CRS)

## Processing Workflow

The extraction process follows these steps:

1. Load the multi-layer raster stack using **Rasterio**.
2. Load the spatial unit polygons using **GeoPandas**.
3. Iterate through each polygon in the GeoDataFrame.
4. Use `rasterio.mask.mask()` to clip the raster according to the polygon geometry.
5. Update raster metadata (size and transform).
6. Export the clipped raster patch as a GeoTIFF file.

During execution, the script prints progress information for each processed spatial unit.

## Dependencies

The script requires the following Python libraries:
rasterio
geopandas


Install them with:

```sql
pip install rasterio geopandas
```sql


## Usage

Run the script from the project root directory:


python extract_spatial_unit_data.py


The script will automatically:

- read the raster and shapefile
- extract raster patches for all spatial units
- store the patches in the output directory

## Notes

- Areas outside the polygon are filled with the raster `nodata` value.
- If the input raster does not define a nodata value, `0` is used as a default.
- The extraction process may take several minutes depending on the number of spatial units and raster resolution.

