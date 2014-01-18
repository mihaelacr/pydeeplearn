""" Disclaimer: this code was adapted from
http://g.sweyla.com/blog/2012/mnist-numpy/
"""

import os, struct
import numpy as np

from array import array as pyarray

"""
Arguments:
Returns:
"""
def read(startExample, count, digits=None, bTrain=True, path="."):
  if digits == None:
    digits = range(0, 10)

  if bTrain:
    fname_img = os.path.join(path, 'train-images-idx3-ubyte')
    fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
  else:
    fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
    fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')

  fImages = open(fname_img,'rb')
  fLabels = open(fname_lbl,'rb')

  # read the header information in the images file.
  s1, s2, s3, s4 = fImages.read(4), fImages.read(4), fImages.read(4), fImages.read(4)
  mnIm = struct.unpack('>I',s1)[0]
  numIm = struct.unpack('>I',s2)[0]
  rowsIm = struct.unpack('>I',s3)[0]
  colsIm = struct.unpack('>I',s4)[0]
  # seek to the image we want to start on
  fImages.seek(16+startExample*rowsIm*colsIm)

  # read the header information in the labels file and seek to position
  # in the file for the image we want to start on.
  mnL = struct.unpack('>I',fLabels.read(4))[0]
  numL = struct.unpack('>I',fLabels.read(4))[0]
  fLabels.seek(8+startExample)

  inputVectors = [] # list of (input, correct label) pairs
  labels = []

  for c in range(count):
    # get the correct label from the labels file.
    val = struct.unpack('>B',fLabels.read(1))[0]
    labels.append(val)

    vec = map(lambda x: struct.unpack('>B',fImages.read(1))[0],
              range(rowsIm*colsIm))
    # get the input from the image file
    inputVectors.append(np.array(vec))


  # Filter out the unwanted digits
  ind = [k for k in xrange(len(labels)) if labels[k] in digits ]
  labels = map(lambda x: labels[x], ind)
  inputVectors = map(lambda x: inputVectors[x], ind)

  fImages.close()
  fLabels.close()

  return np.array(inputVectors), np.array(labels)
