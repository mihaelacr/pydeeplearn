import numpy as np
import utils
from theano.tensor.shared_randomstreams import RandomStreams
from theano import tensor as T
from sklearn import preprocessing
import matplotlib.pyplot as plt


def scale(data):
  return preprocessing.scale(data)

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

def identity(var):
  return var


def getMomentumForEpochLinearIncrease(momentumMax, epoch):
  return np.float32(min(np.float32(0.5) + epoch * np.float32(0.01),
                     np.float32(momentumMax)))

# This is called once per epoch so doing the
# conversion again and again is not a problem
# I do not like this hardcoding business for the GPU: TODO
def getMomentumForEpochSimple(momentumMax, epoch):
  if epoch < 10:
    return np.float32(0.5)
  else:
    return np.float32(momentumMax)


def plotTrainingAndValidationErros(trainingErrors, validationErrors):
  # if run remotely without a display
  try:
    plt.plot(trainingErrors, label="Training error")
    plt.plot(validationErrors, label="Validation error")
    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy average error')
    plt.title('Training and validation error during DBN training')
    plt.legend()
    plt.show()
  except Exception as e:
    print "validation error plot not made"
    print "error ", e

    plt.plot(trainingErrors, label="Training error")
    plt.plot(validationErrors, label="Validation error")
    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy average error')
    plt.title('Training and validation error during DBN training')
    plt.legend()
    plt.savefig("validationandtrainingerror.png" , transparent=True)

    print "printing validation errors and training errors instead"
    print "validationErrors"
    print validationErrors
    print "trainingErrors"
    print trainingErrors

def plotTraningError(trainingErrors):
  try:
    plt.plot(trainingErrors, label="Training error")
    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy average error')
    plt.title('Training error during DBN training')
    plt.legend()
    plt.show()
  except Exception as e:
    print "plot not made"
    print "error ", e

    plt.plot(trainingErrors, label="Training error")
    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy average error')
    plt.title('Training error during DBN training')
    plt.legend()
    plt.savefig("trainingerror.png" , transparent=True)

    print "printing training errors "
    print "trainingErrors"
    print trainingErrors
