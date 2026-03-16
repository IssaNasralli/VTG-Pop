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
   '''
Analysis Tools → Proximity → Create Thiessen Polygons
'''
