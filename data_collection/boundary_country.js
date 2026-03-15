// Define a region of interest for Tunisia
var tunisia = ee.FeatureCollection('FAO/GAUL/2015/level0')
  .filter(ee.Filter.eq('ADM0_NAME', 'Tunisia'));

// Specify the Earth Engine Asset path for export
var exportPath = 'tunisia';
// Add the administrative boundaries to the map
Map.addLayer(tunisia, {color: 'FF0000'}, 'Tunisia Administrative Regions');

// Zoom to Tunisia
Map.centerObject(tunisia, 6);

// Export the dataset to Google Drive
Export.table.toDrive({
  collection: tunisia,
  description: 'tunisia',
  folder: exportPath,
  fileFormat: 'SHP', // Change file format as needed (e.g., 'KML', 'GeoJSON')
});