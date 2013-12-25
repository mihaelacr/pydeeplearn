"""Implementation of restricted boltzmann machine

You need to be able to deal with different energy functions

This allows you to deal with real valued unit

do updates in parallel using multiprocessing.pool

TODO: monitor overfitting
TODO: weight decay (to control overfitting and other things)
TODO: mean filed and dumped mean field

"""
import numpy as np
import math
import multiprocessing

# Global multiprocessing pool, used for all updates in the networks
pool = multiprocessing.Pool()



"""
 Represents a RBM
"""
class RBM(object):
  # 2 layers
  # weights
  # biases for each layer.
  # you can have a type of RBM that just extends the kind of functions
  # you use to update, depending on what kind of energy you use, which also
  # decides what kind of units you have
  # data is a numpy array:
  def __init__(self, data, nrHidden, trainingFunction):
    # Initialize weights to random
    assert len(data) !=0
    self.nrHidden = nrHidden
    self.nrVisible = len(data[0])
    self.weights = initializeWeights(nrVisible, nrHidden)
    self.biases = intializeBiases(data, nrHidden)

  # you need to take a training algorithm as a parameter (CD, PCD)
  def train(self):
    self.weights = trainingFunction(self.data, self.biases, self.weights)

  @classmethod
  def initializeWeights(cls, nrVisible, nrHidden):
    return np.random.normal(0, 0.01, (nrVisible, nrHidden))

  @classmethod
  def initalizeBiases(cls, data, nrHidden):
    # get the procentage of data points that have the i'th unit on
    # and set the visible vias to log (p/(1-p))
    percentages = data.mean(axis=0, dtype='float') / len(data)
    # TODO:what happens if one of them is 1?
    vectorized = np.vectorize(lambda p: math.log(p / (1 -p)))
    visibleBiases = vectorized(percentages)

    # TODO: if sparse hiddeen weights, use that information
    hiddenBiases = np.zeros(nrHidden)
    return visibleBiases, hiddenBiases


# think of adding this to the class
# this might require some inheritance or things
""" Training functions."""
# Makes a step in the contrastiveDivergence algorithm
# online or with mini-bathces?
# you have multiple choices about how to implement this
# It is importaant that the hidden values from the data are binary,
# not probabilities
def contrastiveDivergence(data, biases, weights, cdSteps=1):
  # TODO: do something smarter
  epsilon = 0.0001
  for d in data:
    # TODO: do CDn after some point
    # you can do it by calling the same function with the remaining data
    # TODO: check if you have to use samples
    hidden = updateLayer(Layer.HIDDEN, d, weights, True)
    visibleReconstruction = updateLayer(Layer.VISIBLE, visible, weights, True)
    hiddenReconstruction = updateLayer(Layer.HIDDEN, visible, weights, True)
    weights = weights - epsilon * (np.outer(visible, hidden)
         - np.outer(visibleReconstruction - hiddenReconstruction))

    # TODO: update the biases
  return weights

""" Updates an entire layer. This procedure can be used both in training
    and in testing.
"""
def updateLayer(layer, otherLayerValues, biases, weightMatrix, binary=False):
    bias = biases(layer)

    def activation(x):
      w = getWeights(layer, weightMatrix, x):
      return activationProbability(activationSum(w, bias, otherLayerValues))

  probs = pool.map(activation, xrange(weightMatrix.shape(layer)))

  if binary:
    # Sample from the distributions
    return sampleAll(probs)

def getWeights(layer, weightMatrix, neuronNumber):
  if layer == Layer.VISIBLE:
    return weights[neuronNumber, :]
  else layer == Layer.HIDDEN
    return weights[:, neuronNumber]

# TODO: check if you do it faster with matrix multiplication stuff
# but hinton was adamant about the paralell thing
def activationSum(weights, bias, otherLayerValues):
  return bias + np.dot(weights, otherLayerValues)

""" Gets the activation sums for all the units in one layer.
    Assumesthat the dimensions of the weihgt matrix and biases
    are given correctly. It will throw an exception otherwise.
"""

def activationProbability(activationSum):
  return sigmoid(activationSum)

# Another training algorithm. Slower than Contrastive divergence, but
# gives better results. Not used in practice as it is too slow.
def PCD():
  pass


""" general unitily functions"""

def sigmoid(x):
  return 1 / 1 + np.exp(-x);

def sample(p):
  if np.random.uniform() < p:
    return 1
  return 0

def sampleAll(probs):
  vectorizedSample = np.vectorize(sample)
  return vectorizedSample(probs)

def enum(**enums):
  return type('Enum', (), enums)

# Create an enum for visible and hidden, for
Layer = enum(VISIBLE=0, HIDDEN=1)