"""Implementation of restricted boltzmann machine

You need to be able to deal with different energy functions
This allows you to deal with real valued units.

TODO: monitor overfitting
"""
import numpy as np
from common import *

import theano
from theano import tensor as T
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
    self.cdSteps = theano.shared(value=np.int32(1))
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

    # Create the dropout for the visible layer
    dropoutMask = self.theano_rng.binomial(size=self.visible.shape,
                                          n=1, p=visibleDropout,
                                          dtype=theanoFloat)

    droppedOutVisible = dropoutMask * self.visible
    dropoutMaskHidden = self.theano_rng.binomial(
                              size=(input.shape[0],initialBiases[1].shape[0]),
                              n=1, p=hiddenDropout,
                              dtype=theanoFloat)
    # This does not sample the visible layers, but samples
    # The hidden layers up to the last one, like Hinton suggests
    def OneSampleStep(visibleSample):
      hiddenActivations = T.nnet.sigmoid(T.dot(visibleSample, self.weights) + self.biasHidden)
      hiddenActivationsDropped = hiddenActivations * dropoutMaskHidden
      hidden = self.theano_rng.binomial(size=hiddenActivationsDropped.shape,
                                          n=1, p=hiddenActivationsDropped,
                                          dtype=theanoFloat)

      visibleRec = T.nnet.sigmoid(T.dot(hidden, self.weights.T) + self.biasVisible)
      return [hiddenActivationsDropped, visibleRec]

    results, updates = theano.scan(OneSampleStep,
                          outputs_info=[None, droppedOutVisible],
                          n_steps=self.cdSteps)

    self.updates = updates

    self.hidden = results[0][0]
    self.visibleReconstruction = results[1][-1]

    # Do not sample for the last one, in order to get less sampling noise
    # TODO: drop these as well?
    hiddenRec = T.nnet.sigmoid(T.dot(self.visibleReconstruction, self.weights) + self.biasHidden)
    self.hiddenReconstruction = hiddenRec


# TODO: different learning rates for weights and biases
"""
 Represents a RBM
"""
class RBM(object):

  def __init__(self, nrVisible, nrHidden, learningRate, hiddenDropout,
                visibleDropout, initialWeights=None, initialBiases=None):
    # dropout = 1 means no dropout, keep all the weights
    self.hiddenDropout = hiddenDropout
    # dropout = 1 means no dropout, keep all the weights
    self.visibleDropout = visibleDropout
    self.nrHidden = nrHidden
    self.nrVisible = nrVisible
    self.initialized = False
    self.learningRate = learningRate
    self.weights = initialWeights
    self.biases = initialBiases

  def train(self, data, miniBatchSize=10):
    print "rbm learningRate"
    print self.learningRate

    print "data set size for restricted boltzmann machine"
    print len(data)

    if not self.initialized:
      if self.weights == None and self.biases == None:
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
    cdSteps = T.iscalar()

    batchLearningRate = self.learningRate / miniBatchSize
    batchLearningRate = np.float32(batchLearningRate)

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

    updates.append((batchTrainer.cdSteps, cdSteps))

    train_function = theano.function(
      inputs=[miniBatchIndex, momentum, cdSteps],
      outputs=[], # TODO: output error
      updates=updates,
      givens={
        x: sharedData[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize],
        })

    nrMiniBatches = len(data) / miniBatchSize

    for miniBatchIndex in range(nrMiniBatches):
      if miniBatchIndex < 10:
        momentum = np.float32(0.5)
        step = 1
      else:
        momentum = np.float32(0.95)
        step = 3

      train_function(miniBatchIndex, momentum, step)

    self.weights = batchTrainer.weights.get_value()
    self.biases = [batchTrainer.biasVisible.get_value(),
                   batchTrainer.biasHidden.get_value()]

    self.testWeights = self.weights * self.hiddenDropout

    print "reconstruction Error"
    print reconstructionError(self.biases, self.testWeights, data)

    assert self.weights.shape == (self.nrVisible, self.nrHidden)
    assert self.biases[0].shape[0] == self.nrVisible
    assert self.biases[1].shape[0] == self.nrHidden

  # TODO: move this to GPU as well?
  def reconstruct(self, dataInstances):
    hidden = updateLayer(Layer.HIDDEN, dataInstances, self.biases,
                       self.testWeights, True)
    return updateLayer(Layer.VISIBLE, hidden, self.biases,
                       self.testWeights, False)

""" Updates an entire layer. This procedure can be used both in training
    and in testing.
    Can even take multiple values of the layer, each of them given as rows
    Uses matrix operations.
"""
def updateLayer(layer, otherLayerValues, biases, weights, binary=False):

  bias = biases[layer]
  size = otherLayerValues.shape[0]

  if layer == Layer.VISIBLE:
    activation = np.dot(otherLayerValues, weights.T)
  else:
    activation = np.dot(otherLayerValues, weights)

  probs = sigmoid(np.tile(bias, (size, 1)) + activation)

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

def reconstructionError(biases, weights, data):
    # Returns the rmse of the reconstruction of the data
    # Good to keep track of it, should decrease trough training
    # Initially faster, and then slower
    reconstructions = reconstruct(biases, weights, data)
    return rmse(reconstructions, data)

def reconstruct(biases, weights, dataInstances):
  hidden = updateLayer(Layer.HIDDEN, dataInstances, biases, weights, True)

  visibleReconstructions = updateLayer(Layer.VISIBLE, hidden,
      biases, weights, False)
  return visibleReconstructions