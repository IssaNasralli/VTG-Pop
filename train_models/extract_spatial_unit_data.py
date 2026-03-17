import rasterio
import geopandas as gpd
from rasterio.mask import mask
import os

# -------------------------------
# Input files
# -------------------------------
raster = rasterio.open("tunisia6.tif")
gdf = gpd.read_file("unit_shp/unit_cleaned_filtered.shp")

out_dir = "unit_patches_spyder"
os.makedirs(out_dir, exist_ok=True)

total = len(gdf)
print(f"🚀 Starting extraction of {total} spatial units...")

# -------------------------------
# Iterate over polygons
# -------------------------------
for idx, row in gdf.iterrows():
    geom = [row["geometry"]]

    try:
        # Clip raster to polygon extent
        out_image, out_transform = mask(
            raster,
            geom,
            crop=True,
            filled=False,  # Fill with nodata outside polygon
            nodata=raster.nodata if raster.nodata is not None else 0
        )

        # Update metadata
        out_meta = raster.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Use GeoDataFrame index (idx) as filename
        out_path = os.path.join(out_dir, f"FID_{idx}.tif")

        # Save clipped raster
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(out_image)

        # Progress info
        print(f"[{idx+1}/{total}] ✅ Saved {os.path.basename(out_path)}")

    except Exception as e:
        print(f"[{idx+1}/{total}] ❌ Error with polygon index {idx}: {e}")

print("🎉 Extraction complete! All patches exported.")
