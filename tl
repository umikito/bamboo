/**
 * 最终修正版：按省份循环提交全国竹林分类导出任务
 */

// 1. 定义省份矢量
var provinces = ee.FeatureCollection("FAO/GAUL/2015/level1")
                  .filter(ee.Filter.eq('ADM0_NAME', 'China'));

var year = 2024;
var optBands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];

// ====================== 2. 基础函数定义 ======================

function preprocessS2(img) {
  var scl = img.select('SCL');
  var mask = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7));
  var spectral = img.select(optBands).updateMask(mask);
  var ndvi = spectral.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var evi = spectral.expression('2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))', {
    'B8': spectral.select('B8'), 'B4': spectral.select('B4'), 'B2': spectral.select('B2')
  }).rename('EVI');
  var lswi = spectral.normalizedDifference(['B8', 'B11']).rename('LSWI');
  var mtci = spectral.expression('(B6 - B5) / (B5 - B4)', {
    'B6': spectral.select('B6'), 'B5': spectral.select('B5'), 'B4': spectral.select('B4')
  }).rename('MTCI');
  return spectral.addBands([ndvi, evi, lswi, mtci]).float();
}

function getFinalImage(region) {
  var s2Col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(region).filterDate(year + '-01-01', year + '-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)).map(preprocessS2);
  
  var s2Median = s2Col.median().clip(region);
  var s2Std = s2Col.select(['NDVI', 'MTCI', 'LSWI']).reduce(ee.Reducer.stdDev()).clip(region);
  
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(region).filterDate(year + '-01-01', year + '-12-31')
    .filter(ee.Filter.eq('instrumentMode', 'IW')).median().clip(region);
  var s1Ratio = s1.select('VV').divide(s1.select('VH').add(0.0001)).rename('VV_VH_ratio');
  
  var terrain = ee.Algorithms.Terrain(ee.Image("USGS/SRTMGL1_003")).select(['elevation', 'slope']).clip(region);
  var worldCover = ee.Image("ESA/WorldCover/v100/2020").clip(region);
  var landMask = worldCover.neq(80).and(worldCover.neq(50)).and(worldCover.neq(70));
  
  return s2Median.addBands(s2Std).addBands(s1.select(['VV','VH'])).addBands(s1Ratio).addBands(terrain).updateMask(landMask).float();
}

// ====================== 3. 模型训练 (循环外执行一次) ======================

// 加载训练样本
var forest = ee.FeatureCollection("projects/my-project1-477312/assets/nobanboo_dbs")
  .merge(ee.FeatureCollection("projects/my-project1-477312/assets/forest_samples_ah")).map(function(f){return f.set('class', 0);});
var bamboo = ee.FeatureCollection("projects/my-project1-477312/assets/bamboo_dbs_points")
  .merge(ee.FeatureCollection("projects/my-project1-477312/assets/bamboo_ah")).map(function(f){return f.set('class', 1);});
var grass = ee.FeatureCollection("projects/my-project1-477312/assets/grass_anhui")
  .merge(ee.FeatureCollection("projects/my-project1-477312/assets/grass_samples_ah")).map(function(f){return f.set('class', 2);});

var anhuiSamples_all = forest.merge(bamboo).merge(grass);
var pseudoSamples = ee.FeatureCollection("projects/my-project1-477312/assets/China_Pseudo_Labels_2024");

// 修正点1：统一变量名
var finalTrainingFC = anhuiSamples_all.merge(pseudoSamples);

// 修正点2：优化训练特征提取逻辑。
// 不要裁剪全国影像，而是直接在全量影像集合中针对样本点采样。
var trainingImage = getFinalImage(finalTrainingFC.geometry().bounds());

var trainingData = trainingImage.sampleRegions({
  collection: finalTrainingFC,
  properties: ['class'],
  scale: 10,
  tileScale: 16
}).filter(ee.Filter.notNull(['B2']));

var finalRF = ee.Classifier.smileRandomForest(500).train({
  features: trainingData,
  classProperty: 'class',
  inputProperties: trainingImage.bandNames()
});

// ====================== 4. 循环提交导出任务 ======================

// 获取省份名称列表
var provinceNames = provinces.aggregate_array('ADM1_NAME').getInfo();
var provinceList = provinces.toList(provinces.size());

print('准备提交省份数量：', provinceNames.length);

for (var i = 0; i < provinceNames.length; i++) {
  var name = provinceNames[i].replace(/\s+/g, '_'); 
  var provinceFeat = ee.Feature(provinceList.get(i));
  var provinceGeom = provinceFeat.geometry();
  
  // 为当前省份构建影像
  var provinceImg = getFinalImage(provinceGeom);
  var classified = provinceImg.classify(finalRF);
  
  // 提交导出任务
  Export.image.toDrive({
    image: classified.uint8(),
    description: 'Bamboo_2024_' + name,
    scale: 10,
    region: provinceGeom.bounds(), 
    maxPixels: 1e13,
    folder: 'China_Bamboo_Provinces',
    crs: 'EPSG:4326'
  });
  
  print('任务已排队: ' + name);
}
