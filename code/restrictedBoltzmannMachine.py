"""Implementation of restricted boltzmann machine

You need to be able to deal with different energy functions
This allows you to deal with real valued units.

TODO: monitor overfitting
"""
import numpy as np
from common import *

import theano
from theano import tensor as T
from theano.ifelse import ifelse as theanoifelse
from theano.tensor.shared_randomstreams import RandomStreams

theanoFloat  = theano.config.floatX


EXPENSIVE_CHECKS_ON = False

# I need a mini batch trainer for this

class RBMMiniBatchTrainer(object):

  # TODO: i need to see how I do it with the sampling, because
  # we do not sample all the time to make them binary
  def __init__(self, input, initialWeights, initialBiases,
             visibleDropout, hiddenDropout, cdSteps):

    self.visible = input
    self.cdSteps = cdSteps
    self.theano_rng = RandomStreams(seed=np.random.randint(1, 1000))

    self.weights = theano.shared(value=np.asarray(initialWeights,
                                  dtype=theanoFloat),
                        name='W')
    self.biasVisible = theano.shared(value=np.asarray(initialBiases[0],
                                         dtype=theanoFloat),
                        name='bvis')
    self.biasHidden = theano.shared(value=np.asarray(initialBiases[1],
                                         dtype=theanoFloat),
                        name='bhid')

    oldDw = theano.shared(value=np.zeros(shape=initialWeights.shape,
                                           dtype=theanoFloat))
    oldDVis = theano.shared(value=np.zeros(shape=initialBiases[0].shape,
                                           dtype=theanoFloat))
    oldDHid = theano.shared(value=np.zeros(shape=initialBiases[1].shape,
                                           dtype=theanoFloat))

    self.oldDParams = [oldDw, oldDVis, oldDHid]

    # This does not sample the visible layers, but samples
    # The hidden layers up to the last one, like Hinton suggests
    def OneSampleStep(visibleSample):
      hiddenActivations = T.nnet.sigmoid(T.dot(visibleSample, self.weights) + self.biasHidden)
      hidden = self.theano_rng.binomial(size=hiddenActivations.shape,
                                          n=1, p=hiddenActivations,
                                          dtype=theanoFloat)
      visibleRec = T.nnet.sigmoid(T.dot(hidden, self.weights.T) + self.biasVisible)
      return hidden, visibleRec

    results, updates = theano.scan(OneSampleStep,
                          outputs_info=[None, visibleDropout],
                          n_steps=cdSteps)

    self.updates = updates

    self.hidden = results[0][0]
    self.visibleReconstruction = results[-1][1]

    # Do not sample for the last one, in order to get less sampling noise
    hiddenRec = T.nnet.sigmoid(T.dot(self.visibleReconstruction, self.weights) + self.biasHidden)
    self.hiddenReconstruction = hiddenRec


# TODO: different learning rates for weights and biases
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

  def train(self, data, learningRate=0.01, miniBatchSize=10):
    if not self.initialized:
      self.weights = initializeWeights(self.nrVisible, self.nrHidden)
      self.biases = intializeBiases(data, self.nrHidden)
      self.initialized = True

    sharedData = theano.shared(np.asarray(data, dtype=theanoFloat))

    self.miniBatchSize = miniBatchSize
    # Now you have to build the training function
    # and the updates
    # The mini-batch data is a matrix
    x = T.matrix('x', dtype=theanoFloat)

    miniBatchIndex = T.lscalar()
    momentum = T.fscalar()

    batchLearningRate = learningRate / miniBatchSize
    batchLearningRate = T.cast(batchLearningRate, theanoFloat)

    batchTrainer = RBMMiniBatchTrainer(input=x,
                                       initialWeights=self.weights,
                                       initialBiases=self.biases,
                                       visibleDropout=0.8,
                                       hiddenDropout=0.5,
                                       cdSteps=1)

    updates = []
    # The theano people do not need this because they use gradient
    # I wonder how that works
    positiveDifference = T.dot(batchTrainer.visible.T, batchTrainer.hidden)
    negativeDifference = T.dot(batchTrainer.visibleReconstruction.T,
                               batchTrainer.hiddenReconstruction)
    wUpdate = momentum * batchTrainer.oldDParams[0] + batchLearningRate * (positiveDifference - negativeDifference)
    updates.append((batchTrainer.weights, batchTrainer.weights + wUpdate))
    updates.append((batchTrainer.oldDParams[0], wUpdate))

    visibleBiasDiff = T.sum(x - batchTrainer.visible, axis=0)
    biasVisUpdate = momentum * batchTrainer.oldDParams[1] + batchLearningRate * visibleBiasDiff
    updates.append((batchTrainer.biasVisible, batchTrainer.biasVisible + biasVisUpdate))
    updates.append((batchTrainer.oldDParams[1], biasVisUpdate))


    hiddenBiasDiff = T.sum(batchTrainer.hidden - batchTrainer.hiddenReconstruction, axis=0)
    biasHidUpdate = momentum * batchTrainer.oldDParams[2] + batchLearningRate * hiddenBiasDiff
    updates.append((batchTrainer.biasHidden, batchTrainer.biasHidden + biasHidUpdate))
    updates.append((batchTrainer.oldDParams[2], biasHidUpdate))


    # Add the updates required for the theano random generator
    updates += batchTrainer.updates.items()

    train_function = theano.function(
      inputs=[miniBatchIndex, momentum],
      outputs=[], # TODO: output error
      updates=updates,
      givens={
        x: sharedData[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize],
        })

    nrMiniBatches = len(data) / miniBatchSize
    # The rbm trainign has only one step, you do multiple for the dbn,
    # so maybe not put it here
    epochs = 10
    for epoch in xrange(epochs):
      for miniBatchIndex in range(nrMiniBatches):
        if epoch < 10:
          momentum = 0.5
        else:
          momentum = 0.95

        train_function(miniBatchIndex, momentum)

    self.weights = batchTrainer.weights.get_value()
    self.biases = [batchTrainer.biasVisible.get_value(),
                   batchTrainer.biasHidden.get_value()]

    self.testWeights = self.weights

    print reconstructionError(self.biases, self.weights, data, self.activationFun)


    assert self.weights.shape == (self.nrVisible, self.nrHidden)
    assert self.biases[0].shape[0] == self.nrVisible
    assert self.biases[1].shape[0] == self.nrHidden

  def trainold(self, data):
    # If the network has not been initialized yet, do it now
    # Ie if this is the time it is traning batch of traning
    if not self.initialized:
      self.weights = self.initializeWeights(self.nrVisible, self.nrHidden)
      self.biases = self.intializeBiases(data, self.nrHidden)
      self.initialized = True

    for i in xrange(50):
      self.biases, self.weights = self.trainingFunction(data,
                                                      self.biases,
                                                      self.weights,
                                                      self.activationFun,
                                                      self.dropout,
                                                      self.visibleDropout)
    # TODO:you have to do this for the biases as well
    # TODO: check that I do this in the deep belief net
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

  learningRate = 0.01
  decayFactor = 0.0002
  weightDecay = True
  reconstructionStep = 50

  oldDeltaWeights = np.zeros(weights.shape)
  oldDeltaVisible = np.zeros(biases[0].shape)
  oldDeltaHidden = np.zeros(biases[1].shape)

  batchLearningRate = learningRate / miniBatchSize
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
    # TODO: RMSPROP here as well.
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


def initializeWeights(nrVisible, nrHidden):
  return np.random.normal(0, 0.01, (nrVisible, nrHidden))

def intializeBiases(data, nrHidden):
  # get the procentage of data points that have the i'th unit on
  # and set the visible vias to log (p/(1-p))
  percentages = data.mean(axis=0, dtype='float')
  vectorized = np.vectorize(safeLogFraction, otypes=[np.float])
  visibleBiases = vectorized(percentages)

  hiddenBiases = np.zeros(nrHidden)
  return np.array([visibleBiases, hiddenBiases])
