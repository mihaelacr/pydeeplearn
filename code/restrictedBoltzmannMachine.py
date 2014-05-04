"""Implementation of restricted Boltzmann machine."""

import numpy as np
from common import *

import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

theanoFloat  = theano.config.floatX


class RBMMiniBatchTrainer(object):

  def __init__(self, input, initialWeights, initialBiases,
             visibleActivationFunction, hiddenActivationFunction,
             visibleDropout, hiddenDropout, binary, cdSteps):

    self.visible = input
    self.binary = binary
    self.cdSteps = theano.shared(value=np.int32(cdSteps))
    self.theano_rng = RandomStreams(seed=np.random.randint(1, 1000))

    # Weights and biases
    self.weights = theano.shared(value=np.asarray(initialWeights,
                                  dtype=theanoFloat),
                        name='W')
    self.biasVisible = theano.shared(value=np.asarray(initialBiases[0],
                                         dtype=theanoFloat),
                        name='bvis')
    self.biasHidden = theano.shared(value=np.asarray(initialBiases[1],
                                         dtype=theanoFloat),
                        name='bhid')

    # Old weight and biases updates (required for momentum)
    self.oldDw = theano.shared(value=np.zeros(shape=initialWeights.shape,
                                           dtype=theanoFloat))
    self.oldDVis = theano.shared(value=np.zeros(shape=initialBiases[0].shape,
                                           dtype=theanoFloat))
    self.oldDHid = theano.shared(value=np.zeros(shape=initialBiases[1].shape,
                                           dtype=theanoFloat))

    # Old weight and biases mean squares (required for rmsprop)
    self.oldMeanW = theano.shared(value=np.zeros(shape=initialWeights.shape,
                                           dtype=theanoFloat))
    self.oldMeanVis = theano.shared(value=np.zeros(shape=initialBiases[0].shape,
                                           dtype=theanoFloat))
    self.oldMeanHid = theano.shared(value=np.zeros(shape=initialBiases[1].shape,
                                           dtype=theanoFloat))

    # Create dropout mask for the visible layer
    dropoutMaskVisible = self.theano_rng.binomial(size=self.visible.shape,
                                          n=1, p=visibleDropout,
                                          dtype=theanoFloat)
    # Create dropout mask for the hidden layer
    dropoutMaskHidden = self.theano_rng.binomial(
                              size=(input.shape[0], initialBiases[1].shape[0]),
                              n=1, p=hiddenDropout,
                              dtype=theanoFloat)

    droppedOutVisible = dropoutMaskVisible * self.visible

    # This does not sample the visible layers, but samples
    # The hidden layers up to the last one, like Hinton suggests
    def OneCDStep(visibleSample):
      linearSum = T.dot(visibleSample, self.weights) + self.biasHidden
      hiddenActivations = hiddenActivationFunction(linearSum) * dropoutMaskHidden
      # Sample only for stochastic binary units
      if self.binary:
        hidden = self.theano_rng.binomial(size=hiddenActivations.shape,
                                            n=1, p=hiddenActivations,
                                            dtype=theanoFloat)
      else:
        hidden = hiddenActivations

      linearSum = T.dot(hidden, self.weights.T) + self.biasVisible
      visibleRec = visibleActivationFunction(linearSum) * dropoutMaskVisible
      return [hiddenActivations, visibleRec]

    [hiddenSeq, visibleSeq], updates = theano.scan(OneCDStep,
                          outputs_info=[None, droppedOutVisible],
                          n_steps=self.cdSteps)

    self.updates = updates

    self.hiddenActivations = hiddenSeq[0]
    self.visibleReconstruction = visibleSeq[-1]

    # Do not sample for the last one, in order to get less sampling noise
    hiddenRec = hiddenActivationFunction(T.dot(self.visibleReconstruction, self.weights) + self.biasHidden)
    # TODO: rethink maybe.
    self.hiddenReconstruction = hiddenRec * dropoutMaskHidden


# TODO: give just one theano ring to both the reconstruction batch and the trainer
class ReconstructerBatch(object):
  def __init__(self, input, weights, biases,
             visibleActivationFunction, hiddenActivationFunction,
             visibleDropout, hiddenDropout, binary, cdSteps):

    self.visible = input
    self.binary = binary
    self.cdSteps = theano.shared(value=np.int32(cdSteps))
    self.theano_rng = RandomStreams(seed=np.random.randint(1, 1000))

    self.weightsForVisible, self.weightForHidden = __testWeights(weights,
          visibleDropout=visibleDropout, hiddenDropout=hiddenDropout)

    hiddenBias = biases[1]
    visibleBias = biases[0]
    # This does not sample the visible layers, but samples
    # The hidden layers up to the last one, like Hinton suggests
    def OneCDStep(visibleSample):
      linearSum = T.dot(visibleSample, self.weightForHidden) + hiddenBias
      hiddenActivations = hiddenActivationFunction(linearSum)
      # Sample only for stochastic binary units
      if self.binary:
        hidden = self.theano_rng.binomial(size=hiddenActivations.shape,
                                            n=1, p=hiddenActivations,
                                            dtype=theanoFloat)
      else:
        hidden = hiddenActivations

      linearSum = T.dot(hidden, self.weightsForVisible) + visibleBias
      visibleRec = visibleActivationFunction(linearSum)
      return [hiddenActivations, visibleRec]

    [hiddenSeq, visibleSeq], updates = theano.scan(OneCDStep,
                          outputs_info=[None, self.visible],
                          n_steps=self.cdSteps)

    self.updates = updates

    self.hiddenActivations = hiddenSeq[0]
    self.visibleReconstruction = visibleSeq[-1]

    # Do not sample for the last one, in order to get less sampling noise
    hiddenRec = hiddenActivationFunction(T.dot(self.visibleReconstruction, self.weightForHidden) + hiddenBias)

    self.hiddenReconstruction = hiddenRec

"""
 Represents a RBM
"""
class RBM(object):

  def __init__(self, nrVisible, nrHidden, learningRate,
                hiddenDropout, visibleDropout,
                binary=True,
                visibleActivationFunction=T.nnet.sigmoid,
                hiddenActivationFunction=T.nnet.sigmoid,
                rmsprop=True, nesterov=True,
                weightDecay=0.001,
                initialWeights=None,
                initialBiases=None,
                trainingEpochs=1):
                # TODO: also check how the gradient works for RBMS
    # dropout = 1 means no dropout, keep all the weights
    self.hiddenDropout = hiddenDropout
    # dropout = 1 means no dropout, keep all the weights
    self.visibleDropout = visibleDropout
    self.nrHidden = nrHidden
    self.nrVisible = nrVisible
    self.learningRate = learningRate
    self.rmsprop = rmsprop
    self.nesterov = nesterov
    self.weights = initialWeights
    self.biases = initialBiases
    self.weightDecay = np.float32(weightDecay)
    self.visibleActivationFunction = visibleActivationFunction
    self.hiddenActivationFunction = hiddenActivationFunction
    self.trainingEpochs = trainingEpochs
    self.binary = binary

    self.__initialize(initialWeights, initialBiases)

  def __initialize(self, weights, biases):
    # Initialize the weights
    if weights == None and biases == None:
      weights = initializeWeights(self.nrVisible, self.nrHidden)
      biases = initializeBiasesReal(self.nrVisible, self.nrHidden)
      # if self.binary:
      #   # TODO: I think this makes no sense
      #   self.biases = intializeBiasesBinary(data, self.nrHidden)
      # else:
      #   # TODO: think of this
      #   self.biases = initializeBiasesReal(self.nrVisible, self.nrHidden)

    x = T.matrix('x', dtype=theanoFloat)
    batchTrainer = RBMMiniBatchTrainer(input=x,
                                       initialWeights=weights,
                                       initialBiases=biases,
                                       visibleActivationFunction=self.visibleActivationFunction,
                                       hiddenActivationFunction=self.hiddenActivationFunction,
                                       visibleDropout=self.visibleDropout,
                                       hiddenDropout=self.hiddenDropout,
                                       binary=self.binary,
                                       cdSteps=1)

    reconstructer = ReconstructerBatch(input=x, weights=batchTrainer.weights,
                                        biases=[batchTrainer.biasVisible, batchTrainer.biasHidden],
                                        visibleActivationFunction=self.visibleActivationFunction,
                                        hiddenActivationFunction=self.hiddenActivationFunction,
                                        visibleDropout=self.visibleDropout,
                                        hiddenDropout=self.hiddenDropout,
                                        binary=self.binary,
                                        cdSteps=1)
    self.reconstructer = reconstructer
    self.batchTrainer = batchTrainer
    self.x = x


  def train(self, data, miniBatchSize=10):
    print "rbm learningRate"
    print self.learningRate

    print "data set size for restricted boltzmann machine"
    print len(data)


    sharedData = theano.shared(np.asarray(data, dtype=theanoFloat))

    self.miniBatchSize = miniBatchSize
    # Now you have to build the training function
    # and the updates
    # The mini-batch data is a matrix

    batchTrainer = self.batchTrainer

    miniBatchIndex = T.lscalar()
    momentum = T.fscalar()
    cdSteps = T.iscalar()

    batchLearningRate = self.learningRate / miniBatchSize
    batchLearningRate = np.float32(batchLearningRate)

    if self.nesterov:
      preDeltaUpdates, updates = self.buildNesterovUpdates(batchTrainer,
        momentum, batchLearningRate, cdSteps)
      updateWeightWithMomentum = theano.function(
        inputs=[momentum],
        outputs=[],
        updates=preDeltaUpdates
        )
      updateWeightWithDelta = theano.function(
        inputs=[miniBatchIndex, cdSteps, momentum],
        outputs=[],
        updates=updates,
        givens={
          x: sharedData[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize],
          }
        )

      def trainFunction(miniBatchIndex, momentum, cdSteps):
        updateWeightWithMomentum(momentum)
        updateWeightWithDelta(miniBatchIndex, cdSteps, momentum)

    else:
      updates = self.buildUpdates(batchTrainer, momentum, batchLearningRate, cdSteps)

      trainFunction = theano.function(
        inputs=[miniBatchIndex, momentum, cdSteps],
        outputs=[], # TODO: output error
        updates=updates,
        givens={
          x: sharedData[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize],
          })

    nrMiniBatches = len(data) / miniBatchSize

    for epoch in xrange(self.trainingEpochs):
      for miniBatchIndex in range(nrMiniBatches):
        iteration = miniBatchIndex + epoch * nrMiniBatches
        momentum = np.float32(min(np.float32(0.5) + iteration * np.float32(0.01),
                       np.float32(0.95)))

        if miniBatchIndex < 10:
          step = 1
        else:
          step = 3

        trainFunction(miniBatchIndex, momentum, step)

    self.weights = batchTrainer.weights.get_value()
    self.biases = [batchTrainer.biasVisible.get_value(),
                   batchTrainer.biasHidden.get_value()]

    print "reconstruction Error"
    print self.reconstructionError(data)

    self.testWeights = __testWeights(self.weights, visibleDropout=self.visibleDropout,
                          hiddenDropout=self.hiddenDropout)

    assert self.weights.shape == (self.nrVisible, self.nrHidden)
    assert self.biases[0].shape[0] == self.nrVisible
    assert self.biases[1].shape[0] == self.nrHidden


  def __testWeights(weights, visibleDropout, hiddenDropout):
    return weights.T * hiddenDropout, weights * visibleDropout

  def buildUpdates(self, batchTrainer, momentum, batchLearningRate, cdSteps):
    updates = []
    # The theano people do not need this because they use gradient
    # I wonder how that works
    positiveDifference = T.dot(batchTrainer.visible.T, batchTrainer.hiddenActivations)
    negativeDifference = T.dot(batchTrainer.visibleReconstruction.T,
                               batchTrainer.hiddenReconstruction)
    delta = positiveDifference - negativeDifference

    wUpdate = momentum * batchTrainer.oldDw
    if self.rmsprop:
      meanW = 0.9 * batchTrainer.oldMeanW + 0.1 * delta ** 2
      wUpdate += (1.0 - momentum) * batchLearningRate * delta / T.sqrt(meanW + 1e-8)
      updates.append((batchTrainer.oldMeanW, meanW))
    else:
      wUpdate += (1.0 - momentum) * batchLearningRate * delta

    wUpdate -= batchLearningRate * self.weightDecay * batchTrainer.oldDw

    updates.append((batchTrainer.weights, batchTrainer.weights + wUpdate))
    updates.append((batchTrainer.oldDw, wUpdate))

    visibleBiasDiff = T.sum(batchTrainer.visible - batchTrainer.visibleReconstruction, axis=0)
    biasVisUpdate = momentum * batchTrainer.oldDVis
    if self.rmsprop:
      meanVis = 0.9 * batchTrainer.oldMeanVis + 0.1 * visibleBiasDiff ** 2
      biasVisUpdate += (1.0 - momentum) * batchLearningRate * visibleBiasDiff / T.sqrt(meanVis + 1e-8)
      updates.append((batchTrainer.oldMeanVis, meanVis))
    else:
      biasVisUpdate += (1.0 - momentum) * batchLearningRate * visibleBiasDiff

    updates.append((batchTrainer.biasVisible, batchTrainer.biasVisible + biasVisUpdate))
    updates.append((batchTrainer.oldDVis, biasVisUpdate))

    hiddenBiasDiff = T.sum(batchTrainer.hiddenActivations - batchTrainer.hiddenReconstruction, axis=0)
    biasHidUpdate = momentum * batchTrainer.oldDHid
    if self.rmsprop:
      meanHid = 0.9 * batchTrainer.oldMeanHid + 0.1 * hiddenBiasDiff ** 2
      biasHidUpdate += (1.0 - momentum) * batchLearningRate * hiddenBiasDiff / T.sqrt(meanHid + 1e-8)
      updates.append((batchTrainer.oldMeanHid, meanHid))
    else:
      biasHidUpdate += (1.0 - momentum) * batchLearningRate * hiddenBiasDiff

    updates.append((batchTrainer.biasHidden, batchTrainer.biasHidden + biasHidUpdate))
    updates.append((batchTrainer.oldDHid, biasHidUpdate))

    # Add the updates required for the theano random generator
    updates += batchTrainer.updates.items()

    updates.append((batchTrainer.cdSteps, cdSteps))

    return updates

  def buildNesterovUpdates(self, batchTrainer, momentum, batchLearningRate, cdSteps):
    preDeltaUpdates = []

    wUpdateMomentum = momentum * batchTrainer.oldDw
    biasVisUpdateMomentum = momentum * batchTrainer.oldDVis
    biasHidUpdateMomentum = momentum * batchTrainer.oldDHid

    preDeltaUpdates.append((batchTrainer.weights, batchTrainer.weights + wUpdateMomentum))
    preDeltaUpdates.append((batchTrainer.biasVisible, batchTrainer.biasVisible + biasVisUpdateMomentum))
    preDeltaUpdates.append((batchTrainer.biasHidden, batchTrainer.biasHidden + biasHidUpdateMomentum))

    updates = []
    # The theano people do not need this because they use gradient
    # I wonder how that works, and if it works better
    positiveDifference = T.dot(batchTrainer.visible.T, batchTrainer.hiddenActivations)
    negativeDifference = T.dot(batchTrainer.visibleReconstruction.T,
                               batchTrainer.hiddenReconstruction)
    delta = positiveDifference - negativeDifference
    if self.rmsprop:
      meanW = 0.9 * batchTrainer.oldMeanW + 0.1 * delta ** 2
      wUpdate = (1.0 - momentum) * batchLearningRate * delta / T.sqrt(meanW + 1e-8)
      updates.append((batchTrainer.oldMeanW, meanW))
    else:
      wUpdate = (1.0 - momentum) * batchLearningRate * delta

    wUpdate -= batchLearningRate * self.weightDecay * batchTrainer.oldDw

    updates.append((batchTrainer.weights, batchTrainer.weights + wUpdate))
    updates.append((batchTrainer.oldDw, wUpdate + wUpdateMomentum))

    visibleBiasDiff = T.sum(batchTrainer.visible - batchTrainer.visibleReconstruction, axis=0)
    if self.rmsprop:
      meanVis = 0.9 * batchTrainer.oldMeanVis + 0.1 * visibleBiasDiff ** 2
      biasVisUpdate = (1.0 - momentum) * batchLearningRate * visibleBiasDiff / T.sqrt(meanVis + 1e-8)
      updates.append((batchTrainer.oldMeanVis, meanVis))
    else:
      biasVisUpdate = (1.0 - momentum) * batchLearningRate * visibleBiasDiff

    updates.append((batchTrainer.biasVisible, batchTrainer.biasVisible + biasVisUpdate))
    updates.append((batchTrainer.oldDVis, biasVisUpdate + biasVisUpdateMomentum))

    hiddenBiasDiff = T.sum(batchTrainer.hiddenActivations - batchTrainer.hiddenReconstruction, axis=0)
    if self.rmsprop:
      meanHid = 0.9 * batchTrainer.oldMeanHid + 0.1 * hiddenBiasDiff ** 2
      biasHidUpdate = (1.0 - momentum) * batchLearningRate * hiddenBiasDiff / T.sqrt(meanHid + 1e-8)
      updates.append((batchTrainer.oldMeanHid, meanHid))
    else:
      biasHidUpdate = (1.0 - momentum) * batchLearningRate * hiddenBiasDiff

    updates.append((batchTrainer.biasHidden, batchTrainer.biasHidden + biasHidUpdate))
    updates.append((batchTrainer.oldDHid, biasHidUpdate + biasHidUpdateMomentum))

    # Add the updates required for the theano random generator
    updates += batchTrainer.updates.items()

    updates.append((batchTrainer.cdSteps, cdSteps))

    return preDeltaUpdates, updates

  def hiddenRepresentation(self, dataInstances):
    dataInstacesConverted = theano.shared(np.asarray(dataInstances, dtype=theanoFloat))

    representHidden = theano.function(
            inputs=[],
            outputs=self.reconstructer.hiddenActivations,
            updates=self.reconstructer.updates,
            givens={self.x: dataInstacesConverted})

    return representHidden()

    # TODO: you have to take into account that you are passing in too much
    # data here and it will be too slow
    # so send the data in via mini bathes for reconstruction as well

  def reconstruct(self, dataInstances, cdSteps=1):
    dataInstacesConverted = theano.shared(np.asarray(dataInstances, dtype=theanoFloat))

    reconstructFunction = theano.function(
            inputs=[],
            outputs=self.reconstructer.visibleReconstruction,
            updates=self.reconstructer.updates,
            givens={self.x: dataInstacesConverted})

    return reconstructFunction()

  def reconstructionError(self, dataInstances):
    reconstructions = self.reconstruct(dataInstances)
    return rmse(reconstructions, dataInstances)


def initializeWeights(nrVisible, nrHidden):
  return np.asarray(np.random.normal(0, 0.01, (nrVisible, nrHidden)), dtype=theanoFloat)

# This only works for stochastic binary units
def intializeBiasesBinary(data, nrHidden):
  # get the percentage of data points that have the i'th unit on
  # and set the visible bias to log (p/(1-p))
  percentages = data.mean(axis=0, dtype=theanoFloat)
  vectorized = np.vectorize(safeLogFraction, otypes=[np.float32])
  visibleBiases = vectorized(percentages)

  hiddenBiases = np.zeros(nrHidden, dtype=theanoFloat)
  return np.array([visibleBiases, hiddenBiases])

# TODO: Try random small numbers?
def initializeBiasesReal(nrVisible, nrHidden):
  visibleBiases = np.zeros(nrVisible, dtype=theanoFloat)
  hiddenBiases = np.zeros(nrHidden, dtype=theanoFloat)
  return np.array([visibleBiases, hiddenBiases])
