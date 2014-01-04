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

# can make the thing class methods
# if I feel like I am hardcore, make this callable and replace value with
# the call
class ActivationFunction(object):

  def value(self, input):
    pass

  # why is this not used anywhere? should be  unless it is used explicitely in the
  # functions below
  def derivativeFromValue(self, val):
    pass

  def derivativeForLinearSum(self, topLayerDerivatives, topLayerActivations):
    pass

class Softmax(ActivationFunction):

  def value(self, inputVector):
    expVec = np.vectorize(lambda x: np.exp(x), otypes=[np.float])
    out = expVec(inputVector)
    return out / out.sum()

  def derivativeFromValue(self, value):
    return value * (1.0 - value)

  def derivativeForLinearSum(self, topLayerDerivatives, topLayerActivations):
    # write it as matrix multiplication
    d = - np.outer(topLayerActivations, topLayerActivations)
    d[np.diag_indices_from(d)] = topLayerActivations * (1 - topLayerActivations)
    return np.dot(topLayerDerivatives, d)


class Sigmoid(ActivationFunction):

  def value(self, inputVector):
    return 1 / (1 + np.exp(-inputVector))

  def derivativeFromValue(self, value):
    return value * (1.0 - value)

  def derivativeForLinearSum(self, topLayerDerivatives, topLayerActivations):
    return topLayerActivations * (1 - topLayerActivations) * topLayerDerivatives
