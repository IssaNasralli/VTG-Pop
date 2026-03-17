import geopandas as gpd
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyogrio")

# ============================================================
#  Load input files
# ============================================================
csv_path = "ins.csv"
shp_path = "unit_shp/unit_cleaned.shp"

ins_df = pd.read_csv(csv_path)
gdf = gpd.read_file(shp_path)

# print(f" Loaded {len(ins_df)} rows from {csv_path}")
# print(f" Loaded {len(gdf)} polygons from {shp_path}")

# ============================================================
#  Step 1 — Keep only rows with valid code_sec
# ============================================================
valid_codes = set(gdf["ref_tn_cod"].astype(str))
filtered_ins = ins_df[ins_df["code_sec"].astype(str).isin(valid_codes)].copy()

# print(f" Filtered CSV: {len(ins_df)} → {len(filtered_ins)} rows kept")

# ============================================================
#  Step 2 — Keep only polygons with valid ref_tn_cod
# ============================================================
valid_codes_after = set(filtered_ins["code_sec"].astype(str))
filtered_gdf = gdf[gdf["ref_tn_cod"].astype(str).isin(valid_codes_after)].copy()

# print(f" Filtered Shapefile: {len(gdf)} → {len(filtered_gdf)} polygons kept")

# ============================================================
# 💾 Save results
# ============================================================
filtered_ins.to_csv("ins_filtered.csv", index=False)
filtered_gdf.to_file("unit_shp/unit_cleaned_filtered.shp")

print(" Cleaning completed and files saved:")
print("   - ins_filtered.csv")
print("   - unit_shp/unit_cleaned_filtered.shp")
