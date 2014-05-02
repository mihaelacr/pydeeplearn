"""Implementation of restricted boltzmann machine

You need to be able to deal with different energy functions
This allows you to deal with real valued units.

TODO: monitor overfitting
"""
import numpy as np
from common import *

EXPENSIVE_CHECKS_ON = False

# TODO: different learning rates for weights and biases
# TODO: nesterov method for momentum
# TODO: rmsprop
"""
 Represents a RBM
"""
class RBM(object):

  def __init__(self, nrVisible, nrHidden, trainingFunction, dropout,
                visibleDropout, activationFun=sigmoid):
    # dropout = 1 means no dropout, keep all the weights
    self.dropout = dropout
    # dropout = 1 means no dropout, keep all the weights
    self.visibleDropout = visibleDropout
    self.nrHidden = nrHidden
    self.nrVisible = nrVisible
    self.trainingFunction = trainingFunction
    self.activationFun = activationFun
    self.initialized = False

  def train(self, data):
    # If the network has not been initialized yet, do it now
    # Ie if this is the time it is traning batch of traning
    if not self.initialized:
      self.weights = self.initializeWeights(self.nrVisible, self.nrHidden)
      self.biases = self.intializeBiases(data, self.nrHidden)
      # self.data = data
    # else:
      # self.data = np.concatenate(self.data, data)

    self.biases, self.weights = self.trainingFunction(data,
                                                      self.biases,
                                                      self.weights,
                                                      self.activationFun,
                                                      self.dropout,
                                                      self.visibleDropout)
    self.testWeights = self.weights * self.dropout

    assert self.weights.shape == (self.nrVisible, self.nrHidden)
    assert self.biases[0].shape[0] == self.nrVisible
    assert self.biases[1].shape[0] == self.nrHidden

  """ Reconstructs the data given using this boltzmann machine."""
  def reconstruct(self, dataInstances):
    return reconstruct(self.biases, self.testWeights, dataInstances,
                       self.activationFun)

  def hiddenRepresentation(self, dataInstances):
    return updateLayer(Layer.HIDDEN, dataInstances, self.biases,
                       self.testWeights, self.activationFun, True)

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

    hiddenBiases = np.zeros(nrHidden)
    return np.array([visibleBiases, hiddenBiases])

def reconstruct(biases, weights, dataInstances, activationFun):
  hidden = updateLayer(Layer.HIDDEN, dataInstances, biases, weights,
                       activationFun, True)

  visibleReconstructions = updateLayer(Layer.VISIBLE, hidden,
                                      biases, weights, activationFun, False)
  return visibleReconstructions

def reconstructionError(biases, weights, data, activationFun):
    # Returns the rmse of the reconstruction of the data
    # Good to keep track of it, should decrease trough training
    # Initially faster, and then slower
    reconstructions = reconstruct(biases, weights, data, activationFun)
    return rmse(reconstructions, data)

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
def contrastiveDivergence(data, biases, weights, activationFun, dropout,
                          visibleDropout, miniBatchSize=10):
  N = len(data)
  epochs = N / miniBatchSize

  # sample the probabily distributions allow you to chose from the
  # visible units for dropout
  on = sample(visibleDropout, data.shape)
  dropoutData = data * on

  epsilon = 0.01
  decayFactor = 0.0002
  weightDecay = True
  reconstructionStep = 50

  oldDeltaWeights = np.zeros(weights.shape)
  oldDeltaVisible = np.zeros(biases[0].shape)
  oldDeltaHidden = np.zeros(biases[1].shape)

  batchLearningRate = epsilon / miniBatchSize
  print "batchLearningRate"
  print batchLearningRate

  for epoch in xrange(epochs):
    batchData = dropoutData[epoch * miniBatchSize: (epoch + 1) * miniBatchSize, :]
    if epoch < epochs / 100:
      momentum = 0.5
    else:
      momentum = 0.95

    if epoch < (N/7) * 10:
      cdSteps = 3
    elif epoch < (N/9) * 10:
      cdSteps = 5
    else:
      cdSteps = 10

    if EXPENSIVE_CHECKS_ON:
      if epoch % reconstructionStep == 0:
        print "reconstructionError"
        print reconstructionError(biases, weights, data, activationFun)

    weightsDiff, visibleBiasDiff, hiddenBiasDiff =\
            modelAndDataSampleDiffs(batchData, biases, weights,
            activationFun, dropout, cdSteps)
    # Update the weights
    # data - model
    # Positive phase - negative
    # Weight decay factor
    deltaWeights = (batchLearningRate * weightsDiff
                    - epsilon * weightDecay * decayFactor * weights)

    deltaVisible = batchLearningRate * visibleBiasDiff
    deltaHidden  = batchLearningRate * hiddenBiasDiff

    deltaWeights += momentum * oldDeltaWeights
    deltaVisible += momentum * oldDeltaVisible
    deltaHidden += momentum * oldDeltaHidden

    oldDeltaWeights = deltaWeights
    oldDeltaVisible = deltaVisible
    oldDeltaHidden = deltaHidden

    # Update the weighths
    weights += deltaWeights
    # Update the visible biases
    biases[0] += deltaVisible

    # Update the hidden biases
    biases[1] += deltaHidden

  print reconstructionError(biases, weights, data, activationFun)
  return biases, weights

def modelAndDataSampleDiffs(batchData, biases, weights, activationFun,
                            dropout, cdSteps):
  # Reconstruct the hidden weigs from the data
  hidden = updateLayer(Layer.HIDDEN, batchData, biases, weights, activationFun,
                       binary=True)

  # Chose the units to be active at this point
  # different sets for each element in the mini batches
  on = sample(dropout, hidden.shape)
  dropoutHidden = on * hidden
  hiddenReconstruction = dropoutHidden

  for i in xrange(cdSteps - 1):
    visibleReconstruction = updateLayer(Layer.VISIBLE, hiddenReconstruction,
                                        biases, weights, activationFun,
                                        binary=False)
    hiddenReconstruction = updateLayer(Layer.HIDDEN, visibleReconstruction,
                                       biases, weights, activationFun,
                                       binary=True)
    # sample the hidden units active (for dropout)
    hiddenReconstruction = hiddenReconstruction * on

  # Do the last reconstruction from the probabilities in the last phase
  visibleReconstruction = updateLayer(Layer.VISIBLE, hiddenReconstruction,
                                      biases, weights, activationFun,
                                      binary=False)
  hiddenReconstruction = updateLayer(Layer.HIDDEN, visibleReconstruction,
                                     biases, weights, activationFun,
                                     binary=False)

  hiddenReconstruction = hiddenReconstruction * on
  # here it should be hidden * on - hiddenreconstruction
  # also below in the hidden bias
  weightsDiff = np.dot(batchData.T, dropoutHidden) -\
                np.dot(visibleReconstruction.T, hiddenReconstruction)
  assert weightsDiff.shape == weights.shape

  visibleBiasDiff = np.sum(batchData - visibleReconstruction, axis=0)
  assert visibleBiasDiff.shape == biases[0].shape

  hiddenBiasDiff = np.sum(dropoutHidden - hiddenReconstruction, axis=0)
  assert hiddenBiasDiff.shape == biases[1].shape

  return weightsDiff, visibleBiasDiff, hiddenBiasDiff

""" Updates an entire layer. This procedure can be used both in training
    and in testing.
    Can even take multiple values of the layer, each of them given as rows
    Uses matrix operations.
"""
def updateLayer(layer, otherLayerValues, biases, weights, activationFun,
                binary=False):

  bias = biases[layer]
  size = otherLayerValues.shape[0]

  if layer == Layer.VISIBLE:
    activation = np.dot(otherLayerValues, weights.T)
  else:
    activation = np.dot(otherLayerValues, weights)

  probs = activationFun(np.tile(bias, (size, 1)) + activation)

  if binary:
    # Sample from the distributions
    return sampleAll(probs)

  return probs

# Another training algorithm. Slower than Contrastive divergence, but
# gives better results. Not used in practice as it is too slow.
# This is what Hinton said but it is not OK due to NIPS paper
# This is huge code copy paste but keep it like this for now
def PCD(data, biases, weights, activationFun, dropout,
                          visibleDropout, miniBatchSize=10):
  N = len(data)
  epochs = N / miniBatchSize

  # sample the probabily distributions allow you to chose from the
  # visible units for dropout
  # on = sample(visibleDropout, data.shape)
  # dropoutData = data * on
  dropoutData = data

  epsilon = 0.01
  decayFactor = 0.0002
  weightDecay = True
  reconstructionStep = 50

  oldDeltaWeights = np.zeros(weights.shape)
  oldDeltaVisible = np.zeros(biases[0].shape)
  oldDeltaHidden = np.zeros(biases[1].shape)

  batchLearningRate = epsilon / miniBatchSize
  print "batchLearningRate"
  print batchLearningRate

  # make this an argument or something
  nrFantasyParticles = miniBatchSize

  fantVisible = np.random.randint(2, size=(nrFantasyParticles, weights.shape[0]))
  fantHidden = np.random.randint(2, size=(nrFantasyParticles, weights.shape[1]))

  fantasyParticles = (fantVisible, fantHidden)
  steps = 10

  for epoch in xrange(epochs):
    batchData = dropoutData[epoch * miniBatchSize: (epoch + 1) * miniBatchSize, :]
    if epoch < epochs / 100:
      momentum = 0.5
    else:
      momentum = 0.95

    if EXPENSIVE_CHECKS_ON:
      if epoch % reconstructionStep == 0:
        print "reconstructionError"
        print reconstructionError(biases, weights, data, activationFun)

    print fantasyParticles[0]
    print fantasyParticles[1]
    weightsDiff, visibleBiasDiff, hiddenBiasDiff, fantasyParticles =\
            modelAndDataSampleDiffsPCD(batchData, biases, weights,
            activationFun, dropout, steps, fantasyParticles)

    # Update the weights
    # data - model
    # Positive phase - negative
    # Weight decay factor
    deltaWeights = (batchLearningRate * weightsDiff
                    - epsilon * weightDecay * decayFactor * weights)

    deltaVisible = batchLearningRate * visibleBiasDiff
    deltaHidden  = batchLearningRate * hiddenBiasDiff

    deltaWeights += momentum * oldDeltaWeights
    deltaVisible += momentum * oldDeltaVisible
    deltaHidden += momentum * oldDeltaHidden

    oldDeltaWeights = deltaWeights
    oldDeltaVisible = deltaVisible
    oldDeltaHidden = deltaHidden

    # Update the weighths
    weights += deltaWeights
    # Update the visible biases
    biases[0] += deltaVisible

    # Update the hidden biases
    biases[1] += deltaHidden

  print reconstructionError(biases, weights, data, activationFun)
  return biases, weights


# Same modelAndDataSampleDiff but for persistent contrastive divergence
# First run it without dropout
def modelAndDataSampleDiffsPCD(batchData, biases, weights, activationFun,
                            dropout, steps, fantasyParticles):
  # Reconstruct the hidden weigs from the data
  hidden = updateLayer(Layer.HIDDEN, batchData, biases, weights, activationFun,
                       binary=True)

  # Chose the units to be active at this point
  # different sets for each element in the mini batches
  # on = sample(dropout, hidden.shape)
  # dropoutHidden = on * hidden
  # hiddenReconstruction = dropoutHidden

  for i in xrange(steps):
    visibleReconstruction = updateLayer(Layer.VISIBLE, fantasyParticles[1],
                                        biases, weights, activationFun,
                                        binary=False)
    hiddenReconstruction = updateLayer(Layer.HIDDEN, visibleReconstruction,
                                       biases, weights, activationFun,
                                       binary=True)

    # sample the hidden units active (for dropout)
    # hiddenReconstruction = hiddenReconstruction * on

  fantasyParticles = (visibleReconstruction, hiddenReconstruction)

  # here it should be hidden * on - hiddenReconstruction
  # also below in the hidden bias
  weightsDiff = np.dot(batchData.T, hidden) -\
                np.dot(visibleReconstruction.T, hiddenReconstruction)
  assert weightsDiff.shape == weights.shape

  visibleBiasDiff = np.sum(batchData - visibleReconstruction, axis=0)
  assert visibleBiasDiff.shape == biases[0].shape

  hiddenBiasDiff = np.sum(hidden - hiddenReconstruction, axis=0)
  assert hiddenBiasDiff.shape == biases[1].shape

  return weightsDiff, visibleBiasDiff, hiddenBiasDiff, fantasyParticles