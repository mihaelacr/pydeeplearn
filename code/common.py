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
  out = np.exp(activation)
  return out / out.sum()

def sample(p):
  return np.random.uniform() < p

def sampleAll(probs):
  return np.random.uniform(size=probs.shape) < probs

def enum(**enums):
  return type('Enum', (), enums)

# Create an enum for visible and hidden, for
Layer = enum(VISIBLE=0, HIDDEN=1)

def rmse(prediction, actual):
  return np.linalg.norm(prediction - actual) / np.sqrt(len(prediction))

def safeLogFraction(p):
  assert p >=0 and p <= 1
  # TODO: think about this a bit better
  # you should not set them to be equal, on the contrary,
  # they should be opposites
  if p * (1 - p) == 0:
    return 0
  return np.log(p / (1 -p))

# Takes the value of the sigmoid function and returns the derivative
# Works for numpy arrays as well
def softmaxDerivativeFromVal(valueFunction):
  return valueFunction * (1.0 - valueFunction)

def labelsToVectors(labels, size):
  result = np.zeros((len(labels), size), dtype=float)
  for index, label in enumerate(labels):
    result[index, label] = 1.0

  return result

def zerosFromShape(l):
  return map(lambda x: np.zeros(x.shape), l)

# can make the thing class methods
class ActivationFunction(object):
  pass

""" Implementation of the softmax activation function.
    Used for classification (represents a probablity distribution)
"""
class Softmax(ActivationFunction):

  @staticmethod
  def value(inputVector):
    out = np.exp(inputVector)
    return out / (out.sum(axis=1)[:,None])

  @staticmethod
  def derivativeFromValue(value):
    return value * (1.0 - value)

  @staticmethod
  def derivativeForLinearSum(topLayerDerivatives, topLayerActivations):
    d = - topLayerActivations[:, :, np.newaxis] * topLayerActivations[:, np.newaxis, :]

    vals = topLayerActivations * (1 - topLayerActivations)
    for index in xrange(len(d)):
      d[index][np.diag_indices_from(d[index])] = vals[index]

    res = (topLayerDerivatives[:, :, np.newaxis] * d).sum(axis=1)
    return res


""" Implementation of the sigmoid activation function."""
class Sigmoid(ActivationFunction):

  @staticmethod
  def value(inputVector):
    return 1 / (1 + np.exp(-inputVector))

  @staticmethod
  def derivativeFromValue(value):
    return value * (1.0 - value)

  @staticmethod
  def derivativeForLinearSum(topLayerDerivatives, topLayerActivations):
    return topLayerActivations * (1 - topLayerActivations) * topLayerDerivatives

""" Implementation of the tanh activation function."""
class Tanh(ActivationFunction):

  @staticmethod
  def value(inputVector):
    return np.tanh(inputVector)

  @staticmethod
  def derivativeFromValue(value):
    return 1.0 - value * value

  @staticmethod
  def derivativeForLinearSum(topLayerDerivatives, topLayerActivations):
    return (1.0 - topLayerActivations * topLayerActivations) * topLayerDerivatives

