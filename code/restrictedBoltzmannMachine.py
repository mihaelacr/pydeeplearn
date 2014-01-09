"""Implementation of restricted boltzmann machine

You need to be able to deal with different energy functions

This allows you to deal with real valued unit

do updates in parallel using multiprocessing.pool

TODO: monitor overfitting
TODO: weight decay (to control overfitting and other things)
TODO: mean filed and dumped mean field

"""
import numpy as np
# TODO: work out if you can use this somehow
import multiprocessing

from common import *

EXPENSIVE_CHECKS_ON = False

# TODO: different learning rates for weights and biases
"""
 Represents a RBM
"""
class RBM(object):

  def __init__(self, nrVisible, nrHidden, trainingFunction):
    # Initialize weights to random
    self.nrHidden = nrHidden
    self.nrVisible = nrVisible
    self.trainingFunction = trainingFunction
    self.initialized = False

  def train(self, data):
    # If the network has not been initialized yet, do it now
    # Ie if this is the time it is traning batch of traning
    if not self.initialized:
      self.weights = self.initializeWeights(self.nrVisible, self.nrHidden)
      self.biases = self.intializeBiases(data, self.nrHidden)
      self.data = data
    else:
      self.data = np.concatenate(self.data, data)

    self.biases, self.weights = self.trainingFunction(data,
                                                      self.biases,
                                                      self.weights)
    assert self.weights.shape == (self.nrVisible, self.nrHidden)
    print self.biases[0].shape
    print "nrVisible" + str(self.nrVisible)
    # assert self.biases[0].shape == self.nrVisible

    print self.biases[1].shape
    print "nrHidden" + str(self.nrHidden)
    # assert self.biases[1].shape == self.nrHidden

  """ Reconstructs the data given using this boltzmann machine."""
  def reconstruct(self, dataInstances):
    return reconstruct(self.biases, self.weights, dataInstances)

  def hiddenRepresentation(self, dataInstances):
    return updateLayer(Layer.HIDDEN, dataInstances, self.biases, self.weights, True)

  @classmethod
  def initializeWeights(cls, nrVisible, nrHidden):
    return np.random.normal(0, 0.01, (nrVisible, nrHidden))

  @classmethod
  def intializeBiases(cls, data, nrHidden):
    # get the procentage of data points that have the i'th unit on
    # and set the visible vias to log (p/(1-p))
    percentages = data.mean(axis=0, dtype='float')
    vectorized = np.vectorize(safeLogFraction, otypes=[np.float])
    visibleBiases = vectorized(percentages)

    # TODO: if sparse hiddeen weights, use that information
    hiddenBiases = np.zeros(nrHidden)
    return np.array([visibleBiases, hiddenBiases])

def reconstruct(biases, weights, dataInstances):
  hidden = updateLayer(Layer.HIDDEN, dataInstances, biases, weights, True)

  visibleReconstructions = updateLayer(Layer.VISIBLE, hidden,
                                      biases, weights, False)

  return visibleReconstructions

def reconstructionError(biases, weights, data):
    # Returns the rmse of the reconstruction of the data
    # Good to keep track of it, should decrease trough training
    # Initially faster, and then slower
    recFunc = lambda x: reconstruct(biases, weights, x)
    return rmse(np.array(map(recFunc, data)), data)

""" Training functions."""

""" Full CD function.
Arguments:
  data: the data to use for traning.
    A numpy ndarray.
  biases:

Returns:

Defaults the mini batch size 1, so normal learning
"""
# Think of removing the step method all together and keep one to just
# optimize the code but also make it easier to change them
# rather than have a function  that you pass in for every batch
# if nice and easy refactoring can be seen then you can do that
def contrastiveDivergence(data, biases, weights, miniBatchSize=10):
  N = len(data)

  epochs = N / miniBatchSize

  epsilon = 0.05
  decayFactor = 0.0002
  weightDecay = True
  reconstructionStep = 100

  batchLearningRate = epsilon / miniBatchSize
  print "batchLearningRate"
  print batchLearningRate

  for epoch in xrange(epochs):
    # TODO: you are missing the last part of the data if you
    #
    batchData = data[epoch * miniBatchSize: (epoch + 1) * miniBatchSize, :]
    # TODO: change this and make it proportional to the data
    # like the CD-n
    if epoch < 5:
      momentum = 0.5
    else:
      momentum = 0.9

    if epoch < (N/7) * 10:
      cdSteps = 1
    elif epoch < (N/9) * 10:
      cdSteps = 3
    else:
      cdSteps = 10

    if EXPENSIVE_CHECKS_ON:
      if epoch % reconstructionStep == 0:
        print "reconstructionError"
        print reconstructionError(biases, weights, data)

    weightsDiff, visibleBiasDiff, hiddenBiasDiff =\
            modelAndDataSampleDiffs(batchData, biases, weights)
    # Update the weights
    # data - model
    # Positive phase - negative
    # Weight decay factor
    deltaWeights = (batchLearningRate * weightsDiff
                    - epsilon * weightDecay * decayFactor *  weights)

    deltaVisible = batchLearningRate * visibleBiasDiff
    deltaHidden  = batchLearningRate * hiddenBiasDiff

    # if momentum:
      # this is not required: it is not in Hinton's thing
      # and an if statement might make it considerably shorted in
      # uses in Deep belief networks when we have to train multiple
    if epoch > 1:
      deltaWeights = momentum * oldDeltaWeights + deltaWeights
      deltaVisible = momentum * oldDeltaVisible + deltaVisible
      deltaWeights = momentum * oldDeltaHidden + deltaHidden

    oldDeltaWeights = deltaWeights
    oldDeltaVisible = deltaVisible
    oldDeltaHidden = deltaHidden

    # Update the weighths
    weights += deltaWeights
    # Update the visible biases
    biases[0] += deltaVisible

    # Update the hidden biases
    biases[1] += deltaHidden

  return biases, weights

def modelAndDataSampleDiffs(batchData, biases, weights, cdSteps=1):
  # Reconstruct the hidden weigs from the data
  hidden = updateLayer(Layer.HIDDEN, batchData, biases, weights, True)
  hiddenReconstruction = hidden

  for i in xrange(cdSteps - 1):
    visibleReconstruction = updateLayer(Layer.VISIBLE, hiddenReconstruction,
                                        biases, weights, False)
    hiddenReconstruction = updateLayer(Layer.HIDDEN, visibleReconstruction,
                                       biases, weights, True)

  # Do the last reconstruction from the probabilities in the last phase
  visibleReconstruction = updateLayer(Layer.VISIBLE, hiddenReconstruction,
                                      biases, weights, False)
  hiddenReconstruction = updateLayer(Layer.HIDDEN, visibleReconstruction,
                                     biases, weights, False)

  weightsDiff = np.dot(batchData.T, hidden) - np.dot(visibleReconstruction.T, hiddenReconstruction)
  assert weightsDiff.shape == weights.shape
  visibleBiasDiff = np.sum(batchData - visibleReconstruction, axis=0)

  assert visibleBiasDiff.shape == biases[0].shape
  hiddenBiasDiff = np.sum(hidden - hiddenReconstruction, axis=0)
  assert hiddenBiasDiff.shape == biases[1].shape

  return weightsDiff, visibleBiasDiff, hiddenBiasDiff

# Makes a step in the contrastiveDivergence algorithm
# online or with mini-bathces?
# you have multiple choices about how to implement this
# It is importaant that the hidden values from the data are binary,
# not probabilities

""" Updates an entire layer. This procedure can be used both in training
    and in testing. Does not use matrix multiplication, so it is slower then
    the updateLayer method.
"""
def updateLayerSingle(layer, otherLayerValues, biases, weightMatrix, binary=False):
  bias = biases[layer]

  def activation(x):
    w = weightVectorForNeuron(layer, weightMatrix, x)
    return activationProbability(w, bias[x], otherLayerValues)

  # He said we can update these in parallel but when doing the multibatch that cannot be anymore
  probs = map(activation, xrange(weightMatrix.shape[layer]))
  probs = np.array(probs)

  if binary:
    # Sample from the distributions
    return sampleAll(probs)

  return probs

""" Updates an entire layer. This procedure can be used both in training
    and in testing.
    Can even take multiple values of the layer, each of them given as rows
    Uses matrix operations.
"""
def updateLayer(layer, otherLayerValues, biases, weights, binary=False):
  bias = biases[layer]

  # TODO: think about doing this better
  if len(otherLayerValues.shape) == 2:
    size = otherLayerValues.shape[0]
  else:
    size = 1

  if layer == Layer.VISIBLE:
    activation = np.dot(otherLayerValues, weights.T)
  else:
    activation = np.dot(otherLayerValues, weights)

  probs = sigmoid(np.tile(bias, (size, 1)) + activation)

  if binary:
    # Sample from the distributions
    return sampleAll(probs)

  return probs

"""Function kept in case we go back to trying to make things in parallel
and not with matrix stuff. """
def weightVectorForNeuron(layer, weightMatrix, neuronNumber):
  if layer == Layer.VISIBLE:
    return weightMatrix[neuronNumber, :]
  # else layer == Layer.HIDDEN
  return weightMatrix[:, neuronNumber]

# Made in one function to increase speed
def activationProbability(weights, bias, otherLayerValues):
  return sigmoid(bias + np.dot(weights, otherLayerValues))

def activationSum(weights, bias, otherLayerValues):
  return bias + np.dot(weights, otherLayerValues)

""" Gets the activation sums for all the units in one layer.
    Assumesthat the dimensions of the weihgt matrix and biases
    are given correctly. It will throw an exception otherwise.
"""

# def activationProbability(activationSum):
#   return sigmoid(activationSum)

# Another training algorithm. Slower than Contrastive divergence, but
# gives better results. Not used in practice as it is too slow.
# This is what Hinton said but it is not OK due to NIPS paper
def PCD():
  pass


""" general unitily functions"""

# TODO: move this to common.py
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

