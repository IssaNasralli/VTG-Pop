
import arcpy
import os
from arcpy.sa import *

# Set environment settings
arcpy.env.workspace = "M:\\These\\ThFiLM-Net\\ArcGIS\Assembly\\final_output"
arcpy.env.overwriteOutput = True

# Check out the ArcGIS Spatial Analyst extension license
arcpy.CheckOutExtension("Spatial")

# Input raster
input_raster = "MODIS.tif"

# Function to calculate min and max values of all bands
def get_min_max_values(raster):
    min_val = raster.minimum
    max_val = raster.maximum
    return min_val, max_val

# Function to normalize raster
def normalize_raster(in_raster, out_raster, min_val, max_val):
    normalized_raster = (in_raster - min_val) / (max_val - min_val)
    return normalized_raster

# Load the multi-band raster
multi_band_raster = arcpy.Raster(input_raster)

# Get the minimum and maximum values across all bands
min_val, max_val = get_min_max_values(multi_band_raster)

# Normalize each band based on the minimum and maximum values across all bands
normalized_bands = []

for band_index in range(1, multi_band_raster.bandCount + 1):
    band = arcpy.sa.ExtractBand(input_raster, band_index)
    normalized_band = normalize_raster(band, os.path.join("", f"Normalized_Band_{band_index}.tif"), min_val, max_val)
    normalized_bands.append(normalized_band)

# Combine normalized bands into one multi-band raster
output_raster = "MODIS_Normalized.tif"
arcpy.CompositeBands_management(normalized_bands, output_raster)

print("Normalization complete.")

    
