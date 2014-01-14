""" Disclaimer: this code was taken from:
http://g.sweyla.com/blog/2012/mnist-numpy/
"""
import os, struct
import numpy as np

from array import array as pyarray

def read(digits, dataset = "training", path = "."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in digits ]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.float32)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in xrange(len(ind)):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def readNew(startExample, count, digits, bTrain=True, path="."):
    if not digits:
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

    for blah in range(count):
      val = struct.unpack('>B',fLabels.read(1))[0]
      # get the correct label from the labels file.
      # if val not in digits:
      #   continue

      # issue is here
      labels.append(val)
      # get the input from the image file
      vec = map(lambda x: struct.unpack('>B',fImages.read(1))[0],
                range(rowsIm*colsIm))

      inputVectors.append(np.array(vec))

    ind = [k for k in xrange(len(labels)) if labels[k] in digits ]

    labels = map(lambda x: labels[x], ind)
    inputVectors = map(lambda x: inputVectors[x], ind)

    fImages.close()
    fLabels.close()

    return np.array(inputVectors), np.array(labels)