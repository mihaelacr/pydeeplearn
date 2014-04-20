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

    self.oldMeanW = theano.shared(value=np.zeros(shape=initialWeights.shape,
                                           dtype=theanoFloat))
    self.oldMeanVis = theano.shared(value=np.zeros(shape=initialBiases[0].shape,
                                           dtype=theanoFloat))
    self.oldMeanHid = theano.shared(value=np.zeros(shape=initialBiases[1].shape,
                                           dtype=theanoFloat))

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
                visibleDropout, nesterov=True, initialWeights=None, initialBiases=None):
    # dropout = 1 means no dropout, keep all the weights
    self.hiddenDropout = hiddenDropout
    # dropout = 1 means no dropout, keep all the weights
    self.visibleDropout = visibleDropout
    self.nrHidden = nrHidden
    self.nrVisible = nrVisible
    self.initialized = False
    self.learningRate = learningRate
    self.nesterov = nesterov
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

    if self.nesterov:
      preDeltaUpdates, updates = self.buildNesterovUpdates(batchTrainer,
        momentum, batchLearningRate, cdSteps)
      momentum_function = theano.function(
        inputs=[momentum],
        outputs=[],
        updates=preDeltaUpdates
        )
      after_momentum_updates = theano.function(
        inputs=[miniBatchIndex, cdSteps, momentum],
        outputs=[],
        updates=updates,
        givens={
          x: sharedData[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize],
          }
        )

      def train_function(miniBatchIndex, momentum, cdSteps):
        momentum_function(momentum)
        after_momentum_updates(miniBatchIndex, cdSteps, momentum)

    else:
      updates = self.buildUpdates(batchTrainer, momentum, batchLearningRate, cdSteps)

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
        momentum = np.float32(0.98)
        step = 3

      train_function(miniBatchIndex, momentum, step)

    self.weights = batchTrainer.weights.get_value()
    self.biases = [batchTrainer.biasVisible.get_value(),
                   batchTrainer.biasHidden.get_value()]

    self.testWeights = self.weights * self.hiddenDropout

    print "reconstruction Error"
    print self.reconstructionError(data)

    assert self.weights.shape == (self.nrVisible, self.nrHidden)
    assert self.biases[0].shape[0] == self.nrVisible
    assert self.biases[1].shape[0] == self.nrHidden

  def buildUpdates(self, batchTrainer, momentum, batchLearningRate, cdSteps):
    updates = []
    # The theano people do not need this because they use gradient
    # I wonder how that works
    positiveDifference = T.dot(batchTrainer.visible.T, batchTrainer.hidden)
    negativeDifference = T.dot(batchTrainer.visibleReconstruction.T,
                               batchTrainer.hiddenReconstruction)
    delta = positiveDifference - negativeDifference
    meanW = 0.9 * batchTrainer.oldMeanW + 0.1 * delta ** 2

    wUpdate = momentum * batchTrainer.oldDParams[0]
    wUpdate += batchLearningRate * delta / T.sqrt(meanW + 1e-8)

    updates.append((batchTrainer.weights, batchTrainer.weights + wUpdate))
    updates.append((batchTrainer.oldDParams[0], wUpdate))
    updates.append((batchTrainer.oldMeanW, meanW))

    visibleBiasDiff = T.sum(batchTrainer.visible - batchTrainer.visibleReconstruction, axis=0)
    meanVis = 0.9 * batchTrainer.oldMeanVis + 0.1 * visibleBiasDiff ** 2
    biasVisUpdate = momentum * batchTrainer.oldDParams[1]
    biasVisUpdate += batchLearningRate * visibleBiasDiff / T.sqrt(meanVis + 1e-8)
    updates.append((batchTrainer.biasVisible, batchTrainer.biasVisible + biasVisUpdate))
    updates.append((batchTrainer.oldDParams[1], biasVisUpdate))
    updates.append((batchTrainer.oldMeanVis, meanVis))

    hiddenBiasDiff = T.sum(batchTrainer.hidden - batchTrainer.hiddenReconstruction, axis=0)
    meanHid = 0.9 * batchTrainer.oldMeanHid + 0.1 * hiddenBiasDiff ** 2
    biasHidUpdate = momentum * batchTrainer.oldDParams[2]
    biasHidUpdate += batchLearningRate * hiddenBiasDiff / T.sqrt(meanHid + 1e-8)
    updates.append((batchTrainer.biasHidden, batchTrainer.biasHidden + biasHidUpdate))
    updates.append((batchTrainer.oldDParams[2], biasHidUpdate))
    updates.append((batchTrainer.oldMeanHid, meanHid))

    # Add the updates required for the theano random generator
    updates += batchTrainer.updates.items()

    updates.append((batchTrainer.cdSteps, cdSteps))

    return updates

  def buildNesterovUpdates(self, batchTrainer, momentum, batchLearningRate, cdSteps):
    preDeltaUpdates = []

    wUpdateMomentum = momentum * batchTrainer.oldDParams[0]
    biasVisUpdateMomentum = momentum * batchTrainer.oldDParams[1]
    biasHidUpdateMomentum = momentum * batchTrainer.oldDParams[2]

    preDeltaUpdates.append((batchTrainer.weights, batchTrainer.weights + wUpdateMomentum))
    preDeltaUpdates.append((batchTrainer.biasVisible, batchTrainer.biasVisible + biasVisUpdateMomentum))
    preDeltaUpdates.append((batchTrainer.biasHidden, batchTrainer.biasHidden + biasHidUpdateMomentum))

    updates = []
    # The theano people do not need this because they use gradient
    # I wonder how that works
    positiveDifference = T.dot(batchTrainer.visible.T, batchTrainer.hidden)
    negativeDifference = T.dot(batchTrainer.visibleReconstruction.T,
                               batchTrainer.hiddenReconstruction)
    delta = positiveDifference - negativeDifference
    meanW = 0.9 * batchTrainer.oldMeanW + 0.1 * delta ** 2
    wUpdate = batchLearningRate * delta / T.sqrt(meanW + 1e-8)

    updates.append((batchTrainer.weights, batchTrainer.weights + wUpdate))
    updates.append((batchTrainer.oldDParams[0], wUpdate + wUpdateMomentum))
    updates.append((batchTrainer.oldMeanW, meanW))

    visibleBiasDiff = T.sum(batchTrainer.visible - batchTrainer.visibleReconstruction, axis=0)
    meanVis = 0.9 * batchTrainer.oldMeanVis + 0.1 * visibleBiasDiff ** 2
    biasVisUpdate = batchLearningRate * visibleBiasDiff / T.sqrt(meanVis + 1e-8)

    updates.append((batchTrainer.biasVisible, batchTrainer.biasVisible + biasVisUpdate))
    updates.append((batchTrainer.oldDParams[1], biasVisUpdate + biasVisUpdateMomentum))
    updates.append((batchTrainer.oldMeanVis, meanVis))

    hiddenBiasDiff = T.sum(batchTrainer.hidden - batchTrainer.hiddenReconstruction, axis=0)
    meanHid = 0.9 * batchTrainer.oldMeanHid + 0.1 * hiddenBiasDiff ** 2
    biasHidUpdate = batchLearningRate * hiddenBiasDiff / T.sqrt(meanHid + 1e-8)

    updates.append((batchTrainer.biasHidden, batchTrainer.biasHidden + biasHidUpdate))
    updates.append((batchTrainer.oldDParams[2], biasHidUpdate + biasHidUpdateMomentum))
    updates.append((batchTrainer.oldMeanHid, meanHid))

    # Add the updates required for the theano random generator
    updates += batchTrainer.updates.items()

    updates.append((batchTrainer.cdSteps, cdSteps))

    return preDeltaUpdates, updates

  # TODO: move this to GPU as well?
  # Could be a good idea to speed up things + cleaner
  def hiddenRepresentation(self, dataInstances):
    dataInstacesConverted = np.asarray(dataInstaces, dtype=theanoFloat)

    x = T.matrix('x', dtype=theanoFloat)

    batchTrainer = MiniBatchTrainer(input=x, nrLayers=self.nrLayers,
                                    initialWeights=self.testWeights,
                                    initialBiases=self.biases,
                                    visibleDropout=1,
                                    hiddenDropout=1,
                                    cdSteps=1)

    representHidden = theano.function(
            inputs=[],
            outputs=batchTrainer.hidden,
            updates={},
            givens={x: dataInstacesConverted})

    return representHidden()

    # return lastLayers, np.argmax(lastLayers, axis=1)
    # return updateLayer(Layer.HIDDEN, dataInstances, self.biases,
    #                    self.testWeights, True)

  def reconstruct(self, dataInstances):
    dataInstacesConverted = np.asarray(dataInstances, dtype=theanoFloat)

    x = T.matrix('x', dtype=theanoFloat)

    batchTrainer = MiniBatchTrainer(input=x, nrLayers=self.nrLayers,
                                    initialWeights=self.testWeights,
                                    initialBiases=self.biases,
                                    visibleDropout=1,
                                    hiddenDropout=1,
                                    cdSteps=1)

    reconstruct = theano.function(
            inputs=[],
            outputs=batchTrainer.visibleReconstruction,
            updates={},
            givens={x: dataInstacesConverted})

    return reconstruct()
    # return reconstruct(self.biases, self.testWeights, dataInstances)

  def reconstructionError(self, dataInstances):
    reconstructions = self.reconstruct(dataInstances)
    return rmse(reconstructions, dataInstances)

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
  return np.asarray(np.random.normal(0, 0.01, (nrVisible, nrHidden)), dtype=theanoFloat)

def intializeBiases(data, nrHidden):
  # get the procentage of data points that have the i'th unit on
  # and set the visible vias to log (p/(1-p))
  percentages = data.mean(axis=0, dtype=theanoFloat)
  vectorized = np.vectorize(safeLogFraction, otypes=[np.float32])
  visibleBiases = vectorized(percentages)

  hiddenBiases = np.zeros(nrHidden, dtype=theanoFloat)
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

