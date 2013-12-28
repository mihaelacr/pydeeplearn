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
# TODO: work out if you can use this somehow
import multiprocessing
import logging

# Do not print unless needed
logging.basicConfig(level=logging.CRITICAL)
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
    self.data = data
    self.trainingFunction = trainingFunction
    self.weights = self.initializeWeights(self.nrVisible, self.nrHidden)
    self.biases = self.intializeBiases(data, self.nrHidden)


  # you need to take a training algorithm as a parameter (CD, PCD)
  def train(self):
    self.biases, self.weights = self.trainingFunction(self.data, self.biases, self.weights)

  def reconstruct(self, dataInstance):
    hidden = updateLayer(Layer.HIDDEN, dataInstance, self.biases, self.weights, True)
    visibleReconstruction = updateLayer(Layer.VISIBLE, hidden, self.biases, self.weights, False)
    return visibleReconstruction

  @classmethod
  def initializeWeights(cls, nrVisible, nrHidden):
    return np.random.normal(0, 0.01, (nrVisible, nrHidden))

  @classmethod
  def intializeBiases(cls, data, nrHidden):
    # get the procentage of data points that have the i'th unit on
    # and set the visible vias to log (p/(1-p))
    percentages = data.mean(axis=0, dtype='float') / len(data)
    vectorized = np.vectorize(safeLogFraction)
    visibleBiases = vectorized(percentages)

    # TODO: if sparse hiddeen weights, use that information
    hiddenBiases = np.zeros(nrHidden)
    return np.array([visibleBiases, hiddenBiases])



def safeLogFraction(p):
  assert p >=0 and p <= 1
  # TODO: think about this a bit better
  # you should not set them to be equal, on the contrary, they should be opposites
  if p * (1 - p) == 0:
    return 0
  return math.log(p / (1 -p))

# TODO: add momentum to learning
# TODO: different learning rates for weights and biases

""" Training functions."""

""" Full CD function.
Arguments:
  data: the data to use for traning.
    A numpy ndarray.
  biases:

Returns:
"""
def contrastiveDivergence(data, biases, weights, miniBatch=False):

  N = len(data)
  # Train the first 70 percent of the data with CD1
  endCD1 = math.floor(N / 10 * 7)
  cd1Data = data[0:endCD1, :]
  biases, weights = contrastiveDivergenceStep(cd1Data, biases, weights, cdSteps=1)

  # Train the next 20 percent with CD3
  endCD3 = math.floor(N / 10 * 2) + endCD1
  cd3Data = data[endCD1:endCD3, :]
  biases, weights = contrastiveDivergenceStep(cd3Data, biases, weights, cdSteps=3)

  # Train the next 10 percent of data with CD10
  cd5Data = data[endCD3:N, :]
  biases, weights = contrastiveDivergenceStep(cd5Data, biases, weights, cdSteps=5)

  return biases, weights

# Makes a step in the contrastiveDivergence algorithm
# online or with mini-bathces?
# you have multiple choices about how to implement this
# It is importaant that the hidden values from the data are binary,
# not probabilities
def contrastiveDivergenceStep(data, biases, weights, cdSteps=1):
  # TODO: do something smarter with the learning
  epsilon = 0.0001
  assert cdSteps >=1
  # Check that it does rows in loops
  for visible in data:
    hidden = updateLayer(Layer.HIDDEN, visible, biases, weights, True)
    hiddenReconstruction = hidden
    for i in xrange(cdSteps - 1):
      visibleReconstruction = updateLayer(Layer.VISIBLE, hiddenReconstruction, biases, weights, False)
      hiddenReconstruction = updateLayer(Layer.HIDDEN, visibleReconstruction, biases, weights, True)

    # Do the last reconstruction from the probabilities
    visibleReconstruction = updateLayer(Layer.VISIBLE, hiddenReconstruction, biases, weights, False)
    hiddenReconstruction = updateLayer(Layer.HIDDEN, visibleReconstruction, biases, weights, False)

    weights += epsilon * (np.outer(visible, hidden)
                - np.outer(visibleReconstruction, hiddenReconstruction))

    # Update the visible biases
    biases[0] += epsilon * (visible - visibleReconstruction)
    # Update the hidden biases
    biases[1] += epsilon * (hidden - hiddenReconstruction)
  return biases, weights


""" Updates an entire layer. This procedure can be used both in training
    and in testing.
"""
def updateLayer(layer, otherLayerValues, biases, weightMatrix, binary=False):
  bias = biases[layer]

  # logging.debug("updating layer " + str(layer))
  # logging.debug("with bias" + str(bias))

  # logging.debug("weights" + str(weightMatrix.shape))
  def activation(x):
    # if layer == Layer.VISIBLE:
    #   w =  weightMatrix[x, :]
    # else:
    #   w =  weightMatrix[:, x]

    # return sigmoid(bias[x] + np.dot(w, otherLayerValues))
    w = weightVectorForNeuron(layer, weightMatrix, x)
    return activationProbability(activationSum(w, bias[x], otherLayerValues))

  probs = map(activation, xrange(weightMatrix.shape[layer]))
  probs = np.array(probs)

  if binary:
    # Sample from the distributions
    return sampleAll(probs)

  return probs


def weightVectorForNeuron(layer, weightMatrix, neuronNumber):
  if layer == Layer.VISIBLE:
    return weightMatrix[neuronNumber, :]
  # else layer == Layer.HIDDEN
  return weightMatrix[:, neuronNumber]

# TODO: check if you do it faster with matrix multiplication stuff
# but hinton was adamant about the paralell thing
def activationSum(weights, bias, otherLayerValues):
  # logging.debug("in activationSum")
  # logging.debug(otherLayerValues)
  # logging.debug(weights)

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
  return 1 / (1 + np.exp(-x));

def sample(p):
  return int(np.random.uniform() < p)

def sampleAll(probs):
  return np.random.uniform(size=probs.shape) < probs

def enum(**enums):
  return type('Enum', (), enums)

# Create an enum for visible and hidden, for
Layer = enum(VISIBLE=0, HIDDEN=1)


"""Main. """
def main():
  X = np.array([[1,1,1,0,0,0], # Training subset
                [1,0,1,0,0,0],
                [1,1,1,0,0,0],
                [0,0,1,1,1,0],
                [0,0,1,1,0,0],
                [0,0,1,1,1,0],

                [0,0,1,1,1,1], # Validation subset
                [1,1,1,0,0,0]])
  rbm = RBM(X, 3, contrastiveDivergence)
  rbm.train()


def rmse(prediction, actual):
  return np.linalg.norm(prediction - actual) / np.sqrt(len(prediction))


if __name__ == '__main__':
  main()