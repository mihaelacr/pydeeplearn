import numpy as np

"""
Arguments:
  vec: A numpy 1-D vector.
  size: A 2D tuple

Returns:
  A 2-D vector of dimension 'size', only if 'vec' has compatible dimensions.
  Otherwise it throws an error.
"""
def vectorToImage(vec, size):
  return vec.reshape(size)

""" Transforms the 2D images into 1D vectors
Arguments:
  images: is a python list of numpy arrays
Returns:
  A python list of 1-D numpy arrays, transformed from the input 2D ones
  No data is lost in the transformation.
"""
def imagesToVectors(images):
  return np.array(map(lambda x: x.reshape(-1), images))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(activation):
  expVec = np.vectorize(lambda x: np.exp(x), otypes=[np.float])
  out = expVec(activation)
  return out / out.sum()

# Takes the value of the sigmoid function and returns the derivative
# Works for numpy arrays as well
def softmaxDerivativeFromVal(valueFunction):
  return valueFunction * (1.0 - valueFunction)

def labelsToVectors(labels, size):
  result = np.zeros((len(labels), size), dtype=float)
  for index, label in enumerate(labels):
    result[index, label] = 1

  return result

def indexOfMin(l):
  return min(xrange(len(l)),key=l.__getitem__)