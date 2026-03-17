import os
import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np

print("🔹 Starting program...")

# --- 1. Load shapefile ---
shapefile_path = "unit_shp/unit_cleaned_filtered.shp"
print(f"🔹 Loading shapefile: {shapefile_path}")
gdf = gpd.read_file(shapefile_path)
print(f"✅ Loaded {len(gdf)} polygons")


# --- 3. Compute geometry features ---
print("🔹 Computing geometry features (centroid, area, perimeter)...")
gdf["area_m2"] = gdf.geometry.area
print("✅ Geometry features computed")

# Reset index so the index becomes a column called "FID"
gdf = gdf.reset_index().rename(columns={"index": "FID"})
print("✅ Added FID column")

# --- 4. Build DataFrame with relevant attributes ---
print("🔹 Building dataframe with attributes...")
df = gdf[["FID", "ref_tn_cod", "id_thiesse", "type", "area_m2"]].copy()
print(f"✅ DataFrame built with {len(df)} rows and {df.shape[1]} columns")
df["type"] = df["type"].replace({"rural": 0, "urbain": 1})


# --- 6. Link raster files ---
raster_folder = "unit_patches_spyder"
print(f"🔹 Looking for rasters in: {raster_folder}")
road_means = []
school_means = []

for idx, row in df.iterrows():
    ref_code = str(row["FID"])
    
    # Try to locate the raster file for this unit
    raster_file = None
    for f in os.listdir(raster_folder):
        if f"FID_{ref_code}" in f and f.lower().endswith(".tif"):
            raster_file = os.path.join(raster_folder, f)
            break
    
    if raster_file is None:
        print(f"⚠️ No raster found for unit {ref_code}")
        road_means.append(np.nan)
        school_means.append(np.nan)
        continue
    
    # Read raster bands 7 (road density) and 8 (school KDE)
    try:
        with rasterio.open(raster_file) as src:
            print(f"   📂 Reading raster: {raster_file}")
            band7 = src.read(7).astype(float)  # road density
            band8 = src.read(8).astype(float)  # school KDE
            
            # Mask no-data values
            band7 = band7[band7 != src.nodata] if src.nodata is not None else band7
            band8 = band8[band8 != src.nodata] if src.nodata is not None else band8
            
            road_mean = np.nanmean(band7)
            school_mean = np.nanmean(band8)
            
            road_means.append(road_mean)
            school_means.append(school_mean)
            
            print(f"   ✅ Extracted means → road={road_mean:.4f}, school={school_mean:.4f}")
    except Exception as e:
        print(f"⚠️ Could not read raster {raster_file}: {e}")
        road_means.append(np.nan)
        school_means.append(np.nan)

# --- 7. Add raster-derived features ---
print("🔹 Adding raster-derived features...")
df["road_density_mean"] = road_means
df["school_kde_mean"] = school_means
print("✅ Raster features added")

# --- 8. Replace NaN with 0 ---
print("🔹 Replacing NaN values with 0...")
df = df.fillna({"road_density_mean": 0, "school_kde_mean": 0})
print("✅ Missing values replaced")

# --- 9. Save to CSV ---
output_csv = "spatial_units_features.csv"
print(f"🔹 Saving to CSV: {output_csv}")
df.to_csv(output_csv, index=False)
print(f"✅ CSV saved: {output_csv}")
print("📌 First rows of the final dataframe:")
print(df.head())
