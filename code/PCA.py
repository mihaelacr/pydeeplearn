from scipy import misc

import heapq
import os
import scipy
import scipy.linalg
import numpy
import math
import matplotlib.pyplot as plt

from os.path import isfile, join


# The directory path to the images
PICTURE_PATH = "pics/cambrdige_pics/s1"

# The current directory where the script is ran
currentDir = os.path.dirname(os.path.abspath(__file__))


def convertDataToZeroMean(data):
  means = scipy.mean(data, axis=0)

  # Step2: Substract the mean of it's column from every element
  rows, cols = data.shape
  zeroMean = numpy.zeros((rows, cols))
  for i in xrange(rows):
    zeroMean[i] = data[i] - means

  assert zeroMean.shape == data.shape

  return zeroMean

"""
This method uses the  Karhunen Lowe transform to fastly compute the
eigen vaues of the data.

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
    bigEigVecs[i] = u.T.dot(eigVecs[i])

  # Step 4: Normalize the eigen vectors to get orthonormal components
  bigEigVecs = map(lambda x: x / scipy.linalg.norm(x), bigEigVecs)

  eigValsBigVecs = zip(eigVals, bigEigVecs)
  sortedEigValsBigVecs = sorted(eigValsBigVecs, key=lambda x : x[0], reverse=True)

  index = 0
  result = []
  for eigVal, vector in sortedEigValsBigVecs:
    if index >= dimension:
      break

    if eigVal <=0:
      print "Warning: Non-positive eigen value"

    result += [vector]
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
def pcsWithSVD(train, dimension):
  zeroMean = convertDataToZeroMean(train)

  # SVD guaranteed that the singular values are in non-increasing order
  # this means that the u's are already ordered as required, according
  # to the magnitute of the eigen values
  u, s, vh = scipy.linalg.svd(zeroMean)
  return vh[0:dimension-1]


"""
Arguments:
  vec: A numpy 1-D vector.
  size: A 2D tuple

Returns:
  A 2-D vector of dimension 'size', only if 'vec' has compatible dimensions.
  Otherwise it throws an error.
"""
def transformVectorToImage(vec, size):
  return vec.reshape(size)

""" Transforms the 2D images into 1D vectors
Arguments:
  images: is a python list of numpy arrays
Returns:
  A python list of 1-D numpy arrays, transformed from the input 2D ones
  No data is lost in the transformation.
"""
def trasformImageVectors(images):
  return map(lambda x: x.reshape(-1), images)


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
def getEigenFaces(pcaMethod, images, dimension):
  imgs = map(lambda x: misc.imread(x, flatten=True), images)
  imgSize = imgs[0].shape;
  imgs = trasformImageVectors(imgs)
  imgs = scipy.array(imgs)

  vectors = pcaMethod(imgs, dimension)
  eigenFaces = map(lambda x: transformVectorToImage(x, imgSize), vectors)

  return (eigenFaces, vectors)

def main():
  # Load all the image files in the current directory
  imagePath = os.path.join(currentDir, PICTURE_PATH)
  # TODO: filter only img files
  # TODO: get the images from the other folder as well
  picFiles = [ os.path.join(PICTURE_PATH, f) for f in os.listdir(imagePath)
               if os.path.isfile(os.path.join(imagePath,f)) ]

  eigenFaces, vectors = getEigenFaces(pcsWithSVD, picFiles, 3)
  plt.imshow(eigenFaces[0], cmap=plt.cm.gray)
  plt.show()

  print "done"



if __name__ == '__main__':
  main()