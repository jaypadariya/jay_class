import gdal
import ogr
from sklearn import metrics
from sklearn import svm
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt


#vecto to raster convert
def rasterizeVector(path_to_vector, cols, rows, geo_transform, projection, n_class, raster):
    lblRaster = np.zeros((rows, cols))
    inputDS = ogr.Open(path_to_vector)
    driver = gdal.GetDriverByName('MEM')
    # Define spatial reference(convert vector to raster)
    for j in range(n_class):
        shpLayer = inputDS.GetLayer(0)
        class_id = j + 1
        rasterDS = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
        rasterDS.SetGeoTransform(geo_transform)
        rasterDS.SetProjection(projection)
        shpLayer.SetAttributeFilter("Id = " + str(class_id))
        bnd = rasterDS.GetRasterBand(1)
        bnd.FlushCache()
        gdal.RasterizeLayer(rasterDS, [1], shpLayer, burn_values=[class_id])
        arr = bnd.ReadAsArray()
        lblRaster += arr
        rasterDS = None
        save_raster = gdal.GetDriverByName('GTiff').Create(raster, cols, rows, 1, gdal.GDT_UInt16)
        sband = save_raster.GetRasterBand(1)
        sband.WriteArray(lblRaster)
        sband.FlushCache()
    return lblRaster


def createGeotiff(outRaster, data, geo_transform, projection, dtyp, bcount=1):
    # Create a GeoTIFF file with the given data
    driver = gdal.GetDriverByName('GTiff')
    rows, cols, _ = data.shape
    rasterDS = driver.Create(outRaster, cols, rows, bcount, dtyp)
    rasterDS.SetGeoTransform(geo_transform)
    rasterDS.SetProjection(projection)
    for i in range(bcount):
        band = rasterDS.GetRasterBand(i + 1)
        band.WriteArray(data[:, :, i])
        band.FlushCache()
    return 0


def check_accuracy(actual_labels, predicted_labels, label_count):
    error_matrix = np.zeros((label_count, label_count))
    for actual, predicted in zip(actual_labels, predicted_labels):
        error_matrix[int(actual) - 1][int(predicted) - 1] += 1
    return error_matrix


start = timer()

inpRaster = r"C:\Users\student\Downloads\crop\svm\Input\Image\1.tif"
outRaster = r"C:\Users\student\Downloads\crop\svm\output\SVM.tif"
out_prob = r"C:\Users\student\Downloads\crop\svm\output\svm\Probability_Map.tif"

# Open raster dataset
rasterDS = gdal.Open(inpRaster, gdal.GA_ReadOnly)
# Get spatial reference
geo_transform = rasterDS.GetGeoTransform()
projection = rasterDS.GetProjectionRef()


# Extract band's data and transform into a numpy array
bandsData = []
for b in range(rasterDS.RasterCount):
    band = rasterDS.GetRasterBand(b + 1)
    band_arr = band.ReadAsArray()
    bandsData.append(band_arr)
bandsData = np.dstack(bandsData)
cols, rows, noBands = bandsData.shape

# Read vector data, and rasterize all the vectors in the given directory into a single labelled raster
shapefile = r"C:\Users\student\Downloads\crop\svm\Input\Shapefile\Training_site.shp"
shapefile_test = r"C:\Users\student\Downloads\crop\svm\Input\Shapefile\testing.shp"
rasterized_shp = r"C:\Users\student\Downloads\crop\svm\output\Rasterized.tif"
rasterized_shp_test = r"C:\Users\student\Downloads\crop\svm\output\Rasterized_test.tif"
lblRaster = rasterizeVector(shapefile, rows, cols, geo_transform, projection, n_class=6, raster=rasterized_shp)
lblRaster_test = rasterizeVector(shapefile_test,rows,cols,geo_transform,projection,n_class=6,raster=rasterized_shp_test)

print('Vectors Rasterized to Raster!')

# Prepare training data (set of pixels used for training) and labels
isTrain = np.nonzero(lblRaster)
isTest = np.nonzero(lblRaster_test)
trainingLabels = lblRaster[isTrain]
testingLabels = lblRaster_test[isTest]
trainingData = bandsData[isTrain]
testingData = bandsData[isTest]



# Train SVM Classifier
classifier = svm.SVC(C=100,gamma=0.1,kernel='linear',probability=True,random_state=None,shrinking=True,verbose=False)

classifier.fit(trainingData, trainingLabels)

print('Classifier fitting done!')


# Predict class label of unknown pixels
noSamples = rows * cols
flat_pixels = bandsData.reshape((noSamples, noBands))
result = classifier.predict(flat_pixels)
p_vals = classifier.predict_proba(flat_pixels)
predicted_labels = classifier.predict(trainingData)
lbl_cnt = (np.unique(trainingLabels)).size
df = pd.DataFrame(check_accuracy(trainingLabels, predicted_labels, 6))
df.to_csv(r'C:\Users\student\Downloads\crop\svm\output\CM.csv', index=False)

score_oa = classifier.score(trainingData, trainingLabels)
print('training set OA:', score_oa)
score_oa_test = classifier.score(testingData, testingLabels)
print('testing set OA:', score_oa_test)

predicted_labels_test = classifier.predict(testingData)
test_lbl_cnt = (np.unique(testingLabels)).size

print('Testing Labels: ',np.unique(testingLabels))
print('Predicted Labels: ', np.unique(predicted_labels_test))

df_test = pd.DataFrame(check_accuracy(testingLabels, predicted_labels_test, 6))
df_test.to_csv(r'C:\Users\student\Downloads\crop\svm\output\CM_test.csv', index=False)

print('Confusion Matrices Created!')

###kappa value=======
kappa_score = cohen_kappa_score(trainingLabels, predicted_labels)
print('kappa value training: ', kappa_score)
kappa_score_test = cohen_kappa_score(testingLabels, predicted_labels_test)
print('kappa value testing: ', kappa_score_test)

b_count = p_vals.shape[1]

classification = result.reshape((cols, rows, 1))
prob_arr = p_vals.reshape((cols, rows, b_count))

# Create a GeoTIFF file with the given data
createGeotiff(outRaster, classification, geo_transform, projection, gdal.GDT_UInt16)
createGeotiff(out_prob, prob_arr, geo_transform, projection, gdal.GDT_Float32, b_count)

print('Classified Tiff Image created!')
img = plt.imread(r'C:\Users\student\Downloads\crop\svm\output\SVM.tif')
plt.imshow(img)
plt.show()

end = timer()
print('The process took: ', end - start, ' seconds!')