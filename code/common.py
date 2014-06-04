import numpy as np
import utils
from theano import tensor as T
from sklearn import preprocessing
import matplotlib.pyplot as plt
import itertools


def getClassificationError(predicted, actual):
  return 1.0 - (predicted == actual).sum() * 1.0 / len(actual)

def scale(data):
  return preprocessing.scale(data, axis=1)

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

def shuffle(*args):
  shuffled = shuffleList(*args)
  f = lambda x: np.array(x)
  return tuple(map(f, shuffled))


# Returns lists
def shuffleList(*args):
  lenght = len(args[0])

  # Assert they all have the same size
  assert np.array_equal(np.array(map(len, args)), np.ones(len(args)) * lenght)

  indexShuffle = np.random.permutation(lenght)

  f = lambda x: [x[i] for i in indexShuffle]
  return tuple(map(f, args))

def shuffle3(data1, data2, labels):
  indexShuffle = np.random.permutation(len(data1))
  shuffledData1 = np.array([data1[i] for i in indexShuffle])
  shuffledData2 = np.array([data2[i] for i in indexShuffle])
  shuffledLabels = np.array([labels[i] for i in indexShuffle])

  return shuffledData1, shuffledData2, shuffledLabels


def squaredDiff(first, second):
  return T.sqr(first - second)

# Makes a parameter grid required for cross validation
# the input should be a list of tuples of size 3: min, max and  number of steps
# for each parameter
# EG:  makeParamsGrid([(1, 3, 2), (4,5,2)])
def makeParamsGrid(paramBorders):
  f = lambda x: np.linspace(*x)
  linspaces = map(f, paramBorders)

  return list(itertools.product(*tuple(linspaces)))

# Makes a parameter grid required for cross validation
# the input should be a list of tuples of size 3: min, max and  number of steps
# for each parameter
# EG:  makeParamsGrid([(1, 3, 2), (4,5,2)])
def makeParamsGrid(paramBorders):
  f = lambda x: np.linspace(*x)
  linspaces = map(f, paramBorders)

  return list(itertools.product(*tuple(linspaces)))


def getMomentumForEpochLinearIncrease(momentumMax, epoch):
  return np.float32(min(np.float32(0.5) + epoch * np.float32(0.1),
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

    # If we had an error we are either not sshed with -X
    # or we are in a detached screen session.
    # so turn the io off and save the pic
    plt.ioff()
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

    plt.ioff()
    plt.plot(trainingErrors, label="Training error")
    plt.xlabel('Epoch')
    plt.ylabel('Cross entropy average error')
    plt.title('Training error during DBN training')
    plt.legend()
    plt.savefig("trainingerror.png" , transparent=True)

    print "printing training errors "
    print "trainingErrors"
    print trainingErrors


def plot3Errors(trainingErrors, trainWithDropout, validationErrors):
  # if run remotely without a display
  plt.plot(trainWithDropout, label="Training error on dropped out set.")
  plt.plot(trainingErrors, label="Training error")
  plt.plot(validationErrors, label="Validation error")
  plt.xlabel('Epoch')
  plt.ylabel('Cross entropy average error')
  plt.title('Training and validation error during DBN training')
  plt.legend()
  plt.show()
