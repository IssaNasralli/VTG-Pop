// Define a function to collect MODIS images for a given region and year
var CollectMODISForRegionAndYear = function(regionName, level, admName, startYear, endYear) {
  // Define a region of interest (ROI) based on the specified parameters
  var country = ee.FeatureCollection('FAO/GAUL/2015/' + level)
    .filter(ee.Filter.eq(admName, regionName));
  var roi = country.geometry();

  // Create an empty image collection to store MODIS images
  var modisCollection = ee.ImageCollection([]);

  // Iterate over the years
  for (var year = startYear; year <= endYear; year++) {
    // Set the start and end dates based on the year parameter
    var startDate = year + '-01-01';
    var endDate = year + '-12-31';

    // Load MODIS data
    var dataset = ee.ImageCollection('MODIS/061/MCD43A4')
                  .filter(ee.Filter.date(startDate, endDate))
                  .select([
                    'Nadir_Reflectance_Band1', 'Nadir_Reflectance_Band4',
                    'Nadir_Reflectance_Band3'
                  ])
                  .filterBounds(roi); // Apply ROI filter
    
    // Add the dataset to the collection
    modisCollection = modisCollection.merge(dataset);
  }

  // Create a composite image representing the median statistic over the time period
  var medianComposite = modisCollection.median();
  
  // Visualization parameters
  var trueColorVis = {
    min: 0.0,
    max: 4000.0,
    gamma: 1.4,
  };

  // Set map center and add layer to the map
  Map.centerObject(roi, 6); // Center the map on the ROI and set zoom level
  Map.addLayer(medianComposite, trueColorVis, 'MODIS Median Composite - ' + regionName); // Set last parameter to false to indicate no stretching
  Map.addLayer(roi, { color: 'FF0000' }, 'Region of Interest'); // Add ROI to the map
  
  // Export the median composite image to Google Drive
  Export.image.toDrive({
    image: medianComposite,
    description: 'MODIS_Median_Composite_' + regionName,
    folder: 'MODIS_Images', // Specify the folder in Google Drive
    fileNamePrefix: 'modis_median_composite_' + regionName,
    scale: 500,
    region: roi, // Set the region of interest
    crs: 'EPSG:32632',
    maxPixels: 1e13
  });
};

// Call the function with the desired parameters
CollectMODISForRegionAndYear('Tunisia', 'level0', 'ADM0_NAME', 2024, 2024);