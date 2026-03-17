import geopandas as gpd

# Load shapefile
gdf = gpd.read_file("unit.shp")

# Normalize to string for safe comparison
gdf['ref_tn_cod'] = gdf['ref_tn_cod'].astype(str).str.strip()

# Count before filtering
before = len(gdf)

# Drop rows where ref_tn_cod is empty, NaN, or "0"
gdf = gdf[~gdf['ref_tn_cod'].isin(["", "nan", "NaN", "None", "0"])]

# Drop actual NaN values too
gdf = gdf.dropna(subset=['ref_tn_cod'])

after = len(gdf)

print(f"✅ Removed {before - after} polygons without ref_tn_cod")

# Save cleaned shapefile
gdf.to_file("unit_cleaned.shp")

print("Saved cleaned shapefile as unit_cleaned.shp")
