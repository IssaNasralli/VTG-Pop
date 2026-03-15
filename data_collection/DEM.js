// Load the GAUL dataset
var gaul = ee.FeatureCollection("FAO/GAUL/2015/level0");

// Filter the GAUL dataset for Tunisia
var tunisia = gaul.filter(ee.Filter.eq('ADM0_NAME', 'Tunisia'));

// Load the SRTM DEM dataset
var srtm = ee.Image("USGS/SRTMGL1_003");

// Clip the DEM to the Tunisia boundary
var demTunisia = srtm.clip(tunisia);

// Display the DEM on the map
Map.centerObject(tunisia, 7);
Map.addLayer(demTunisia, {min: 0, max: 3000, palette: ['blue', 'green', 'yellow', 'brown', 'white']}, 'SRTM DEM Tunisia');

// Calculate the slope from the DEM
var slopeTunisia = ee.Terrain.slope(demTunisia);

// Display the slope on the map
Map.addLayer(slopeTunisia, {min: 0, max: 60, palette: ['00FFFF', '008000', 'FFFF00', 'FFA500', 'FF0000']}, 'Slope Tunisia');

// Export the DEM layer to Google Drive with higher maxPixels
Export.image.toDrive({
  image: demTunisia,
  description: 'SRTM_DEM_Tunisia',
  scale: 30,
  region: tunisia.geometry(),
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9  // Increase the maxPixels to handle large area
});

// Export the Slope layer to Google Drive with higher maxPixels
Export.image.toDrive({
  image: slopeTunisia,
  description: 'SRTM_Slope_Tunisia',
  scale: 30,
  region: tunisia.geometry(),
  fileFormat: 'GeoTIFF',
  maxPixels: 1e9  // Increase the maxPixels to handle large area
});