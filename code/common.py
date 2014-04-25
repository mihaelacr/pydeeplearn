import numpy as np
import utils
from theano.tensor.shared_randomstreams import RandomStreams
from theano import tensor as T


def visualizeWeights(weights, imgShape, tileShape):
  return utils.tile_raster_images(weights, imgShape,
                                  tileShape, tile_spacing=(1, 1))
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

# Do not use theano's softmax, it is numerically unstable
# and it causes Nans to appear
# Semantically this is the same
def softmax(v):
  e_x = T.exp(v - v.max(axis=1, keepdims=True))
  return e_x / e_x.sum(axis=1, keepdims=True)

def sample(p, size):
  return np.random.uniform(size=size) <= p

# this can be done with a binomial
def sampleAll(probs):
  return np.random.uniform(size=probs.shape) <= probs

def enum(**enums):
  return type('Enum', (), enums)

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


def labelsToVectors(labels, size):
  result = np.zeros((len(labels), size), dtype=float)
  for index, label in enumerate(labels):
    result[index, label] = 1.0

  return result

def zerosFromShape(l):
  return map(lambda x: np.zeros(x.shape), l)

def shuffle(data, labels):
  indexShuffle = np.random.permutation(len(data))
  shuffledData = np.array([data[i] for i in indexShuffle])
  shuffledLabels = np.array([labels[i] for i in indexShuffle])

  return shuffledData, shuffledLabels

# Recitified linear unit
def relu(var):
  return var * (var > 0.0)

def cappedRelu(var):
  return var * (var > 0.0) * (var < 6.0)

def noisyRelu(var, theano_rng):
  # TODO: might have to add the gaussain noise
  var += theano_rng.normal(avg=0.0, std=1.0)
  return var * (var > 0.0)

def makeNoisyRelu():
  rng = RandomStreams(seed=np.random.randint(1, 1000))

  return lambda var: noisyRelu(var, rng)

def noisyReluSigmoid(var, theano_rng):
  var += theano_rng.normal(avg=0.0, std=T.nnet.ultra_fast_sigmoid(var))
  return var * (var > 0.0)

def makeNoisyReluSigmoid():
  rng = RandomStreams(seed=np.random.randint(1, 1000))

  return lambda var: noisyReluSigmoid(var, rng)

def id(var):
  return var