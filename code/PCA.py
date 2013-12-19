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
    Numpy array of arrays?

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

def main():
  # Load all the image files in the current directory
  imagePath = os.path.join(currentDir, PICTURE_PATH)
  # TODO: filter only img files
  # TODO: get the images from the other folder as well
  picFiles = [ os.path.join(PICTURE_PATH, f) for f in os.listdir(imagePath)
               if os.path.isfile(os.path.join(imagePath,f)) ]

  imgs = map(lambda x: misc.imread(x, flatten=True), picFiles)
  imgSize = imgs[0].shape;
  imgs = trasformImageVectors(imgs)
  imgs = scipy.array(imgs)
  result = pca(imgs, 3)

  imagePcas = map(lambda x: transformVectorToImage(x, imgSize), result)
  plt.imshow(imagePcas[0], cmap=plt.cm.gray)
  plt.show()

  print "done"



if __name__ == '__main__':
  main()