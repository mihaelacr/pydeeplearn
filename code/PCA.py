from scipy import misc

import heapq
import os
import scipy
import scipy.linalg
import numpy
import math
import matplotlib.pyplot as plt

from os.path import isfile, join

PICTURE_PATH = "pics/cambrdige_pics/s1"

currentDir = os.path.dirname(os.path.abspath(__file__))

"""
Arguments:
  train:
    Numpy array of arrays
Returns:
  The principal components of the data.

"""
# Returns the principal components of the given training
# data by commputing the principal eigen vectors of the
# covariance matrix of the data
def pca(train, dimension):
  # Use the Karhunen Lowe transform to fastly compute
  # the principal components.

  # Step1: Get the mean of each column of the data
  # Ie create the average image
  means = scipy.mean(train, axis=0)

  # Step2: Substract the mean of it's column from every element
  rows, cols = train.shape
  u = numpy.zeros((rows, cols))
  for i in xrange(rows):
    u[i] = train[i] - means

  assert u.shape == train.shape

  # Step3: Compute the eigen values of the U * U^T matrix
  # the size of U * U^T is rows * rows (ie the number of data points you have
  # in your training)
  eigVals, eigVecs = scipy.linalg.eig(u.dot(u.T))

  # Step4: Compute the eigen values of U^T*U from the eigen values of U * U^T
  bigEigVecs = numpy.zeros((rows, cols))
  for i in xrange(rows):
    bigEigVecs[i] = u.T.dot(eigVecs[i])

  # Step 5: Normalize the eigen vectors to get orthonormal components
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


# TODO: remove code duplication between this and above
def pcsWithSVD(train, dimension):
  # Step1: Get the mean of each column of the data
  # Ie create the average image
  means = scipy.mean(train, axis=0)

  # Step2: Substract the mean of it's column from every element
  rows, cols = train.shape
  zeroMean = numpy.zeros((rows, cols))
  for i in xrange(rows):
    zeroMean[i] = train[i] - means

  assert zeroMean.shape == train.shape

  u, s, vh = scipy.linalg.svd(zeroMean)

  print s
  print vh.shape
  return vh[0:dimension-1]


"""
Arguments:
  vec:
  size: A 2D tuple
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
  images: A python list of images that have to be of the same size.
Returns:
  A tuple:
    The first element of the tuple is formed from the eigen faces of given
      images.
    The second element of the tuple if formed from the vector version of the
      eigen faces. This is kept for optimization reasons.
"""
def getEigenFaces(images, dimension):
  imgs = map(lambda x: misc.imread(x, flatten=True), images)
  imgSize = imgs[0].shape;
  imgs = trasformImageVectors(imgs)
  imgs = scipy.array(imgs)

  vectors = pcsWithSVD(imgs, dimension)
  eigenFaces = map(lambda x: transformVectorToImage(x, imgSize), vectors)

  return (eigenFaces, vectors)

def main():
  # Load all the image files in the current directory
  imagePath = os.path.join(currentDir, PICTURE_PATH)
  # TODO: filter only img files
  # TODO: get the images from the other folder as well
  picFiles = [ os.path.join(PICTURE_PATH, f) for f in os.listdir(imagePath)
               if os.path.isfile(os.path.join(imagePath,f)) ]

  eigenFaces, vectors = getEigenFaces(picFiles, 3)
  plt.imshow(eigenFaces[0], cmap=plt.cm.gray)
  plt.show()

  print "done"



if __name__ == '__main__':
  main()