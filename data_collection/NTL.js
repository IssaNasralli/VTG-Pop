// Define a function to collect Nighttime Lights images for a given region and year
var CollectNighttimeLightsForRegionAndYear = function(regionName, level, admName, startYear, endYear) {
  // Define a region of interest (ROI) based on the specified parameters
  var country = ee.FeatureCollection('FAO/GAUL/2015/' + level)
    .filter(ee.Filter.eq(admName, regionName));
  var geometry = country.geometry();

  // Iterate over the years
  for (var year = startYear; year <= endYear; year++) {
    // Set the start and end dates based on the year parameter
    var startDate = year + '-01-01';
    var endDate = year + '-12-31';

    // Load VIIRS Nighttime Lights data for the specified year
    var dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
      .filter(ee.Filter.date(startDate, endDate))
      .filterBounds(geometry);
    var nighttime = dataset.select('avg_rad');
    var nighttimeVis = {min: 0.0, max: 60.0};

    // Calculate the median composite
    var medianComposite = nighttime.median();

    // Display Nighttime Lights on the map
    Map.centerObject(geometry, 6);
    Map.addLayer(medianComposite, nighttimeVis, 'Nighttime Lights - ' + regionName + ' ' + year);
    Map.addLayer(geometry, {color: 'FF0000'}, 'Country Boundary');

    // Adjust region name for export folder if necessary
    if (regionName === 'Sfax') {
      regionName = 'sfax';
    }

    // Export the image to Google Drive
    Export.image.toDrive({
      image: medianComposite,
      description: 'Nighttime_Lights_' + regionName + '_' + year,
      folder: 'Nighttime_Lights_Images', // Specify the folder in Google Drive
      fileNamePrefix: 'Nighttime_Lights_' + regionName + '_' + year,
      scale: 463,
      region: geometry,
      crs: 'EPSG:32632',
      maxPixels: 1e13
    });
  }
};

CollectNighttimeLightsForRegionAndYear('Tunisia', 'level0', 'ADM0_NAME', 2024, 2024);