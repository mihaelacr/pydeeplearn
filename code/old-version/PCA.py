import heapq
import matplotlib.pyplot as plt
import numpy
import os
import scipy
import scipy.linalg

from os.path import isfile, join
from scipy import misc

# Import all common functions
from common import *


# The directory path to the images
PICTURE_PATH = "/pics/cambrdige_pics/"

# The current directory where the script is ran
currentDir = os.path.dirname(os.path.abspath(__file__))

"""
  Converts the data to zero mean data.
"""
def convertDataToZeroMean(data):
  means = scipy.mean(data, axis=0)
  rows, cols = data.shape
  zeroMean = numpy.zeros((rows, cols))
  for i in xrange(rows):
    zeroMean[i] = data[i] - means

  assert zeroMean.shape == data.shape

  return zeroMean


"""
  Uses a heuristic to evaluate how many dimensions should the data be reduced
    to.

Arguments:
  eigenValues:
    The eigen values of the covariance matrix, or numbers proportional to them.
    Should be a numpy 1-D array.
Returns:
  The dimension the data should be reduced to.
"""
def dimensionFromEigenIndividualVariance(eigenValues):
  threshold = 0.01
  dimension = 0

  s = numpy.sum(eigenValues)
  print "sum eigen" + str(s)

  for eigen in eigenValues:
    r = eigen / s
    if r > threshold:
      dimension += 1

  return dimension

# requires the eigen values to be sorted before
def dimensionFromEigenTotalVariance(eigenValues):
  threshold = 0.95
  dimension = 0

  s = numpy.sum(eigenValues)
  print "sum eigen" + str(s)
  current = 0
  for eigen in eigenValues:
    r = (eigen / s)
    current += r
    if current >= threshold:
      break
    dimension += 1

  return dimension


"""
This method uses the  Karhunen Lowe transform to fastly compute the
eigen vaues of the data.

It is faster than the SVD method below, but can be prone to floating point
errors more than the SVD one.
Arguments:
  train:
    Numpy array of arrays
  dimension: the dimension to which to reduce the size of the data set.

Returns:
  The principal components of the data.
"""
# Returns the principal components of the given training
# data by commputing the principal eigen vectors of the
# covariance matrix of the data
def pca(train, dimension):
  # Use the Karhunen Lowe transform to fastly compute
  # the principal components.
  rows, cols = train.shape
  # Step1: Get the mean of each column of the data
  # Ie create the average image
  u = convertDataToZeroMean(train)

  # Step2: Compute the eigen values of the U * U^T matrix
  # the size of U * U^T is rows * rows (ie the number of data points you have
  # in your training)
  eigVals, eigVecs = scipy.linalg.eig(u.dot(u.T))


  # Step3: Compute the eigen values of U^T*U from the eigen values of U * U^T
  bigEigVecs = numpy.zeros((rows, cols))
  for i in xrange(rows):
    bigEigVecs[i] = u.T.dot(eigVecs[:, i])

  # Step 4: Normalize the eigen vectors to get orthonormal components
  bigEigVecs = map(lambda x: x / scipy.linalg.norm(x), bigEigVecs)

  eigValsBigVecs = zip(eigVals, bigEigVecs)
  sortedEigValsBigVecs = sorted(eigValsBigVecs, key=lambda x : x[0], reverse=True)

  index = 0
  if dimension == None:
    # Get the eigen values
    # Note that these are not the eigen values of the covariance matrix
    # but the eigen values of U * U ^T
    # however, this is fine because they just differ by a factor
    # so the ratio between eigen values will be preserved
    eigenValues = map(lambda x : x[0], sortedEigValsBigVecs)
    dimension = dimensionFromEigenTotalVariance(eigenValues)
    print "Using PCA dimension " + str(dimension)


  result = np.empty(rows, dimension)
  for eigVal, vector in sortedEigValsBigVecs:
    if index >= dimension:
      break

    if eigVal <=0:
      print "Warning: Non-positive eigen value"

    result[:, index] = vector
    index = index + 1

  return result

"""
Arguments:
  train:
    Numpy array of arrays
  dimension: the dimension to which to reduce the size of the data set.

Returns:
  The principal components of the data.

  This method should be preferred over the above: it is well known that the
  SVD methods are more stable than the ones that require the computation of
  the eigen values and eigen vectors.
  For more detail see:
  http://math.stackexchange.com/questions/3869/what-is-the-intuitive-relationship-between-svd-and-pca
"""
def pcaWithSVD(train, dimension=None):
  zeroMean = convertDataToZeroMean(train)

  # SVD guaranteed that the singular values are in non-increasing order
  # this means that the u's are already ordered as required, according
  # to the magnitute of the eigen values
  u, s, vh = scipy.linalg.svd(zeroMean)

  if dimension == None:
    # Get the eigen values from the singular values
    eigenValues = s ** 2;
    dimension = dimensionFromEigenTotalVariance(eigenValues)
    print "Using PCA dimension " + str(dimension)

  return vh[0:dimension-1]

"""
Arguments:
  pcaMethod: a method to use for PCA.
  images: A python list of images that have to be of the same size.
  dimension: the dimension to which to reduce the size of the data set.
Returns:
  A tuple:
    The first element of the tuple is formed from the eigen faces of given
      images.
    The second element of the tuple if formed from the vector version of the
      eigen faces. This is kept for optimization reasons.
"""
def getEigenFaces(pcaMethod, images, dimension=None):

  imgSize = images[0].shape;
  # this call should not be here: the code should assume that the images have
  # been transofrmed to vectors before
  imgs = imagesToVectors(images)

  vectors = pcaMethod(imgs, dimension)
  eigenFaces = map(lambda x: vectorToImage(x, imgSize), vectors)

  return (eigenFaces, vectors)


def reduce(principalComponents, vectors):
  assert len(principalComponents) > 0

  print principalComponents[0].shape

  principalComponents = np.array(principalComponents)

  lowDimRepresentation  = np.dot(vectors, principalComponents.T)
  # lowDimRepresentation = map(lambda x : vectors.dot(x), principalComponents)
  # sameDimRepresentation = \
  #   sum([ x * y for x, y in zip(principalComponents, lowDimRepresentation)])
  # TODO: do this with einsum
  sameDimRepresentation = lowDimRepresentation[:, np.newaxis] * principalComponents.T
  sameDimRepresentation = sameDimRepresentation.sum(axis=2)
  # TODO: create the proper thing here so that you can
  # easily see what the ouput is
  return  (lowDimRepresentation, sameDimRepresentation)


"""
 Reduces a 2D image represented by a numpy 2D array of integer values(pixels)
 to a lower dimension, dictated by the number of principal components.
"""
def reduceImageToLowerDimensions(principalComponents, image2D):
  assert len(principalComponents) > 0

  size = principalComponents[0].shape
  vector = vectorToImage(image2D, size)

  lowDimRepresentation = map(lambda x : x.T.dot(vector), principalComponents)
  sameDimRepresentation = \
    sum([ x * y for x, y in zip(principalComponents, lowDimRepresentation)])
  return  (lowDimRepresentation, sameDimRepresentation)


def main():
  # Load all the image files in the current directory
  picFiles = []
  path = currentDir + PICTURE_PATH
  for root, dirs, files in os.walk(path):
    if root != path:
      picFiles += map(lambda x: os.path.join(root, x), files)

  print len(picFiles)

  imgs = map(lambda x: misc.imread(x, flatten=True), picFiles)

  eigenFaces, principalComponents = getEigenFaces(pca, imgs)
  # plt.imshow(eigenFaces[0], cmap=plt.cm.gray)
  # plt.show()

  lowDimRepresentation, sameDimRepresentation = \
      reduceImageToLowerDimensions(principalComponents, imgs[0])

  plt.imshow(imgs[0], cmap=plt.cm.gray)
  plt.show()

  image2D = vectorToImage(sameDimRepresentation, imgs[0].shape)
  plt.imshow(image2D, cmap=plt.cm.gray)
  plt.show()
  print "done"



if __name__ == '__main__':
  main()