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

EXPENSIVE_CHECKS_ON = False

# Global multiprocessing pool, used for all updates in the networks
pool = multiprocessing.Pool()


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

  def reconstruct(self, dataInstance):
    return reconstruct(self.biases, self.weights, dataInstance)

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


def reconstruct(biases, weights, dataInstance):
  hidden = updateLayer(Layer.HIDDEN, dataInstance, biases, weights, True)
  visibleReconstruction = updateLayer(Layer.VISIBLE, hidden,
                                      biases, weights, False)
  return visibleReconstruction

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
def contrastiveDivergence(data, biases, weights, miniBatchSize=1):
  N = len(data)

  epochs = N / miniBatchSize

  epsilon = 0.001
  decayFactor = 0.0002
  weightDecay = True

  for epoch in xrange(epochs):
    # TODO: you are missing the last part of the data if you
    #
    batchData = data[epoch * miniBatchSize: (epoch + 1) * miniBatchSize]
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

    # Move the reconstruction at the end
    for i in xrange(len(batchData)):
    #   if EXPENSIVE_CHECKS_ON:
    #     if i % reconstructionStep == 0:
    #       print "reconstructionError"
    #       print reconstructionError(biases, weights, data)

      weightsDiff, visibleBiasDiff, hiddenBiasDiff = modelAndDataSampleDiffs(batchData[i], biases, weights)
      # Update the weights
      # data - model
      # Positive phase - negative
      # Weight decay factor
      deltaWeights = (epsilon / len(batchData) * weightsDiff
                      - epsilon * weightDecay * decayFactor *  weights)

      deltaVisible = epsilon * visibleBiasDiff
      deltaHidden  = epsilon * hiddenBiasDiff

      # if momentum:
        # this is not required: it is not in Hinton's thing
        # and an if statement might make it considerably shorted in
        # uses in Deep belief networks when we have to train multiple
      if i > 1:
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


def modelAndDataSampleDiffs(visible, biases, weights, cdSteps=1):
  # Reconstruct the hidden weigs from the data
  hidden = updateLayer(Layer.HIDDEN, visible, biases, weights, True)
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

  weightsDiff = np.outer(visible, hidden) - np.outer(visibleReconstruction, hiddenReconstruction)
  visibleBiasDiff = visible - visibleReconstruction
  hiddenBiasDiff = hidden - hiddenReconstruction

  return weightsDiff, visibleBiasDiff, hiddenBiasDiff

# Makes a step in the contrastiveDivergence algorithm
# online or with mini-bathces?
# you have multiple choices about how to implement this
# It is importaant that the hidden values from the data are binary,
# not probabilities
"""

  Momentum and weight decay should always be on for performance. s
  TODO: Understand more why momentum works. And why it should be done like this
  TODO: for performance might consider doing two if statements and repeating the code?
  or just remove the option
"""
def contrastiveDivergenceStep(data, biases, weights, cdSteps=1, momentum=True, weightDecay=True):
  # TODO: do something smarter with the learning
  epsilon = 0.001
  decayFactor = 0.0002
  momentum = 0.5
  assert cdSteps >=1

  N = len(data)

  # How often should you compute the reconstruction error of the data
  reconstructionStep = N / 100

  # TODO: try and rewrite some of these things to use matrix stuff
  # but then you lose the chance of doing parallel stuff
  for i in xrange(N):
    if EXPENSIVE_CHECKS_ON:
      if i % reconstructionStep == 0:
        print "reconstructionError"
        print reconstructionError(biases, weights, data)

    visible = data[i]
    # Reconstruct the hidden weigs from the data
    hidden = updateLayer(Layer.HIDDEN, visible, biases, weights, True)
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

    # Update the weights
    # Positive phase
    deltaWeights = epsilon * (np.outer(visible, hidden)
                    # Negative phase
                    -  np.outer(visibleReconstruction, hiddenReconstruction)
                    # Weight decay factor
                    - weightDecay * decayFactor *  weights)

    deltaVisible = epsilon * (visible - visibleReconstruction)
    deltaHidden  = epsilon * (hidden - hiddenReconstruction)

    # TODO: do first step differently
    if momentum:

      # this is not required: it is not in Hinton's thing
      # and an if statement might make it considerably shorted in
      # uses in Deep belief networks when we have to train multiple
      if i > 1:
        deltaWeights = momentum * oldDeltaWeights + deltaWeights
        deltaVisible = momentum * oldDeltaVisible + deltaVisible
        deltaWeights = momentum * oldDeltaHidden + deltaHidden

      oldDeltaWeights = deltaWeights
      oldDeltaVisible = deltaVisible
      oldDeltaHidden = deltaHidden

    weights += deltaWeights
    # Update the visible biases
    biases[0] += deltaVisible

    # Update the hidden biases
    biases[1] += deltaHidden

  return biases, weights


""" Updates an entire layer. This procedure can be used both in training
    and in testing.
"""
def updateLayer(layer, otherLayerValues, biases, weightMatrix, binary=False):
  bias = biases[layer]

  def activation(x):
    w = weightVectorForNeuron(layer, weightMatrix, x)
    return activationProbability(w, bias[x], otherLayerValues)

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

def rmse(prediction, actual):
  return np.linalg.norm(prediction - actual) / np.sqrt(len(prediction))

def safeLogFraction(p):
  assert p >=0 and p <= 1
  # TODO: think about this a bit better
  # you should not set them to be equal, on the contrary,
  # they should be opposites
  if p * (1 - p) == 0:
    return 0
  return math.log(p / (1 -p))

