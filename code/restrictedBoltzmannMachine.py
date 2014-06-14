"""Implementation of restricted Boltzmann machine."""

import numpy as np
from common import *

from activationfunctions import *

import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

theanoFloat  = theano.config.floatX


class RBMMiniBatchTrainer(object):

  def __init__(self, input, theanoGenerator, initialWeights, initialBiases,
             visibleActivationFunction, hiddenActivationFunction,
             visibleDropout, hiddenDropout, sparsityConstraint, cdSteps):

    self.visible = input
    self.cdSteps = theano.shared(value=np.int32(cdSteps))
    self.theanoGenerator = theanoGenerator

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

    if visibleDropout in [1.0, 1]:
      droppedOutVisible = self.visible
    else:
      # Create dropout mask for the visible layer
      dropoutMaskVisible = self.theanoGenerator.binomial(size=self.visible.shape,
                                            n=1, p=visibleDropout,
                                            dtype=theanoFloat)

      droppedOutVisible = dropoutMaskVisible * self.visible


    if hiddenDropout in [1.0, 1]:
      dropoutMaskHidden = T.ones(shape=(input.shape[0], initialBiases[1].shape[0]))
    else:
      # Create dropout mask for the hidden layer
      dropoutMaskHidden = self.theanoGenerator.binomial(
                                size=(input.shape[0], initialBiases[1].shape[0]),
                                n=1, p=hiddenDropout,
                                dtype=theanoFloat)

    def OneCDStep(visibleSample):
      linearSum = T.dot(visibleSample, self.weights) + self.biasHidden
      hidden = hiddenActivationFunction.nonDeterminstic(linearSum) * dropoutMaskHidden

      linearSum = T.dot(hidden, self.weights.T) + self.biasVisible
      if visibleDropout in [1.0, 1]:
        visibleRec = visibleActivationFunction.deterministic(linearSum)
      else:
        visibleRec = visibleActivationFunction.deterministic(linearSum) * dropoutMaskVisible

      return visibleRec

    visibleSeq, updates = theano.scan(OneCDStep,
                          outputs_info=[droppedOutVisible],
                          n_steps=self.cdSteps)

    self.updates = updates

    self.visibleReconstruction = visibleSeq[-1]

    self.runningAvgExpected = theano.shared(value=np.zeros(shape=initialBiases[1].shape,
                                           dtype=theanoFloat))

    # Duplicate work but avoiding gradient in theano thinking we are using a random op
    linearSum = T.dot(droppedOutVisible, self.weights) + self.biasHidden
    self.hiddenActivations = hiddenActivationFunction.deterministic(linearSum) * dropoutMaskHidden

    self.activationProbabilities =  hiddenActivationFunction.activationProbablity(linearSum)

    # Do not sample for the last one, in order to get less sampling noise
    # Here you should also use a expected value for symmetry
    # but we need an elegant way to do it
    hiddenRec = hiddenActivationFunction.deterministic(T.dot(self.visibleReconstruction, self.weights) + self.biasHidden)
    self.hiddenReconstruction = hiddenRec * dropoutMaskHidden


# TODO: check if this is doing the right thing
# because the graph makes it look like hidden activations has to do with
# the random generator
class ReconstructerBatch(object):
  def __init__(self, input, theanoGenerator, weights, biases,
             visibleActivationFunction, hiddenActivationFunction,
             visibleDropout, hiddenDropout, cdSteps):

    self.visible = input
    self.cdSteps = theano.shared(value=np.int32(cdSteps))
    self.theanoGenerator = theanoGenerator

    self.weightsForVisible, self.weightsForHidden = testWeights(weights,
          visibleDropout=visibleDropout, hiddenDropout=hiddenDropout)

    hiddenBias = biases[1]
    visibleBias = biases[0]

    # This does not sample the visible layers, but samples
    # The hidden layers up to the last one, like Hinton suggests
    def OneCDStep(visibleSample):
      linearSum = T.dot(visibleSample, self.weightsForHidden) + hiddenBias
      hidden = hiddenActivationFunction.nonDeterminstic(linearSum)
      linearSum = T.dot(hidden, self.weightsForVisible) + visibleBias
      visibleRec = visibleActivationFunction.deterministic(linearSum)

      return visibleRec

    visibleSeq, updates = theano.scan(OneCDStep,
                          outputs_info=[self.visible],
                          n_steps=self.cdSteps)

    self.updates = updates

    self.visibleReconstruction = visibleSeq[-1]

    # Duplicate work but avoiding gradient in theano thinking we are using a random op
    linearSum = T.dot(self.visible, self.weightsForHidden) + hiddenBias
    self.hiddenActivations = hiddenActivationFunction.deterministic(linearSum)

    # Do not sample for the last one, in order to get less sampling noise
    hiddenRec = hiddenActivationFunction.deterministic(T.dot(self.visibleReconstruction, self.weightsForHidden) + hiddenBias)

    self.hiddenReconstruction = hiddenRec

"""
 Represents a RBM
"""
class RBM(object):

  def __init__(self, nrVisible, nrHidden, learningRate,
                hiddenDropout, visibleDropout,
                visibleActivationFunction=Sigmoid(),
                hiddenActivationFunction=Sigmoid(),
                rmsprop=True,
                nesterov=True,
                weightDecay=0.001,
                initialWeights=None,
                initialBiases=None,
                trainingEpochs=1,
                momentumFactorForLearningRate=False,
                momentumMax=0.95,
                sparsityCostFunction=T.nnet.binary_crossentropy,
                sparsityConstraint=False,
                sparsityRegularization=0.01,
                sparsityTraget=0.01):
                # TODO: also check how the gradient works for RBMS
    # dropout = 1 means no dropout, keep all the weights
    self.hiddenDropout = hiddenDropout
    print "hidden dropout in RBM" , hiddenDropout
    # dropout = 1 means no dropout, keep all the weights
    self.visibleDropout = visibleDropout
    print "visible dropout in RBM" , visibleDropout
    self.nrHidden = nrHidden
    self.nrVisible = nrVisible
    self.learningRate = learningRate
    self.rmsprop = rmsprop
    self.nesterov = nesterov
    self.weights = initialWeights
    self.biases = initialBiases
    self.weightDecay = np.float32(weightDecay)
    self.momentumFactorForLearningRate = momentumFactorForLearningRate
    self.visibleActivationFunction = visibleActivationFunction
    self.hiddenActivationFunction = hiddenActivationFunction
    self.trainingEpochs = trainingEpochs
    self.momentumMax = momentumMax
    self.sparsityConstraint = sparsityConstraint
    self.sparsityRegularization = np.float32(sparsityRegularization)
    self.sparsityTraget = np.float32(sparsityTraget)
    self.sparsityCostFunction = sparsityCostFunction


    if sparsityConstraint:
      print "using sparsityConstraint"

    self.__initialize(initialWeights, initialBiases)

  def __initialize(self, weights, biases):
    # Initialize the weights
    if weights == None and biases == None:
      weights = initializeWeights(self.nrVisible, self.nrHidden)
      biases = initializeBiasesReal(self.nrVisible, self.nrHidden)

    theanoRng = RandomStreams(seed=np.random.randint(1, 1000))

    x = T.matrix('x', dtype=theanoFloat)
    batchTrainer = RBMMiniBatchTrainer(input=x,
                                       theanoGenerator=theanoRng,
                                       initialWeights=weights,
                                       initialBiases=biases,
                                       visibleActivationFunction=self.visibleActivationFunction,
                                       hiddenActivationFunction=self.hiddenActivationFunction,
                                       visibleDropout=self.visibleDropout,
                                       hiddenDropout=self.hiddenDropout,
                                       sparsityConstraint=self.sparsityConstraint,
                                       cdSteps=1)

    reconstructer = ReconstructerBatch(input=x,
                                        theanoGenerator=theanoRng,
                                        weights=batchTrainer.weights,
                                        biases=[batchTrainer.biasVisible, batchTrainer.biasHidden],
                                        visibleActivationFunction=self.visibleActivationFunction,
                                        hiddenActivationFunction=self.hiddenActivationFunction,
                                        visibleDropout=self.visibleDropout,
                                        hiddenDropout=self.hiddenDropout,
                                        cdSteps=1)
    self.reconstructer = reconstructer
    self.batchTrainer = batchTrainer
    self.x = x


  def train(self, data, miniBatchSize=10):
    print "rbm learningRate"
    print self.learningRate

    print "data set size for restricted boltzmann machine"

    print len(data)

    # TODO: bring it back if it helps
    # If we have gaussian units, we need to scale the data
    # to unit variance and zero mean
    if isinstance(self.visibleActivationFunction, Identity):
      print "scaling data for RBM"
      data = scale(data)

    sharedData = theano.shared(np.asarray(data, dtype=theanoFloat))

    self.miniBatchSize = miniBatchSize

    batchTrainer = self.batchTrainer
    x = self.x

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
      print "rbm training epoch", epoch
      for miniBatchIndex in range(nrMiniBatches):
        iteration = miniBatchIndex + epoch * nrMiniBatches
        momentum = np.float32(min(np.float32(0.5) + iteration * np.float32(0.001),
                       np.float32(self.momentumMax)))

        if miniBatchIndex < 10:
          step = 1
        else:
          step = 3

        trainFunction(miniBatchIndex, momentum, step)

    self.sharedWeights = batchTrainer.weights
    self.sharedBiases = [batchTrainer.biasVisible, batchTrainer.biasHidden]
    self.weights = batchTrainer.weights.get_value()
    self.biases = [batchTrainer.biasVisible.get_value(),
                   batchTrainer.biasHidden.get_value()]

    print "reconstruction Error"
    print self.reconstructionError(data)

    self.testWeights = testWeights(self.weights, visibleDropout=self.visibleDropout,
                          hiddenDropout=self.hiddenDropout)

    assert self.weights.shape == (self.nrVisible, self.nrHidden)
    assert self.biases[0].shape[0] == self.nrVisible
    assert self.biases[1].shape[0] == self.nrHidden


  def buildUpdates(self, batchTrainer, momentum, batchLearningRate, cdSteps):
    updates = []

    if self.momentumFactorForLearningRate:
      factorLr = 1.0 - momentum
    else:
      factorLr = np.float32(1.0)

    if self.sparsityConstraint:
      if self.sparsityCostFunction == T.nnet.binary_crossentropy:
        sparistyCostMeasure = batchTrainer.activationProbabilities
      else:
        sparistyCostMeasure = batchTrainer.hiddenActivations

      runningAvg = batchTrainer.runningAvgExpected * 0.9 + T.mean(sparistyCostMeasure, axis=0) * 0.1
      # Sum over all hidden units
      sparsityCost = T.sum(self.sparsityCostFunction(self.sparsityTraget, runningAvg))

      updates.append((batchTrainer.runningAvgExpected, runningAvg))

    positiveDifference = T.dot(batchTrainer.visible.T, batchTrainer.hiddenActivations)
    negativeDifference = T.dot(batchTrainer.visibleReconstruction.T,
                               batchTrainer.hiddenReconstruction)
    delta = positiveDifference - negativeDifference

    wUpdate = momentum * batchTrainer.oldDw

    # # Sparsity cost
    # if self.sparsityConstraint:
    #   gradientW = T.grad(sparsityCost, batchTrainer.weights)
    #   delta -= self.sparsityRegularization * gradientW

    if self.rmsprop:
      meanW = 0.9 * batchTrainer.oldMeanW + 0.1 * delta ** 2
      wUpdate += factorLr * batchLearningRate * delta / T.sqrt(meanW + 1e-8)
      updates.append((batchTrainer.oldMeanW, meanW))
    else:
      wUpdate += factorLr * batchLearningRate * delta

    wUpdate -= batchLearningRate * self.weightDecay * batchTrainer.oldDw


    updates.append((batchTrainer.weights, batchTrainer.weights + wUpdate))
    updates.append((batchTrainer.oldDw, wUpdate))

    visibleBiasDiff = T.sum(batchTrainer.visible - batchTrainer.visibleReconstruction, axis=0)
    biasVisUpdate = momentum * batchTrainer.oldDVis

    if self.rmsprop:
      meanVis = 0.9 * batchTrainer.oldMeanVis + 0.1 * visibleBiasDiff ** 2
      biasVisUpdate += factorLr * batchLearningRate * visibleBiasDiff / T.sqrt(meanVis + 1e-8)
      updates.append((batchTrainer.oldMeanVis, meanVis))
    else:
      biasVisUpdate += factorLr * batchLearningRate * visibleBiasDiff

    updates.append((batchTrainer.biasVisible, batchTrainer.biasVisible + biasVisUpdate))
    updates.append((batchTrainer.oldDVis, biasVisUpdate))

    hiddenBiasDiff = T.sum(batchTrainer.hiddenActivations - batchTrainer.hiddenReconstruction, axis=0)
    biasHidUpdate = momentum * batchTrainer.oldDHid

     # Sparsity cost
    if self.sparsityConstraint:
      gradientbiasHid = T.grad(sparsityCost, batchTrainer.biasHidden)
      hiddenBiasDiff -= self.sparsityRegularization * gradientbiasHid

    if self.rmsprop:
      meanHid = 0.9 * batchTrainer.oldMeanHid + 0.1 * hiddenBiasDiff ** 2
      biasHidUpdate += factorLr * batchLearningRate * hiddenBiasDiff / T.sqrt(meanHid + 1e-8)
      updates.append((batchTrainer.oldMeanHid, meanHid))
    else:
      biasHidUpdate += factorLr * batchLearningRate * hiddenBiasDiff

    updates.append((batchTrainer.biasHidden, batchTrainer.biasHidden + biasHidUpdate))
    updates.append((batchTrainer.oldDHid, biasHidUpdate))

    # Add the updates required for the theano random generator
    updates += batchTrainer.updates.items()

    updates.append((batchTrainer.cdSteps, cdSteps))

    return updates

  def buildNesterovUpdates(self, batchTrainer, momentum, batchLearningRate, cdSteps):
    preDeltaUpdates = []


    if self.momentumFactorForLearningRate:
      factorLr = 1.0 - momentum
    else:
      factorLr = np.float32(1.0)

    wUpdateMomentum = momentum * batchTrainer.oldDw
    biasVisUpdateMomentum = momentum * batchTrainer.oldDVis
    biasHidUpdateMomentum = momentum * batchTrainer.oldDHid

    preDeltaUpdates.append((batchTrainer.weights, batchTrainer.weights + wUpdateMomentum))
    preDeltaUpdates.append((batchTrainer.biasVisible, batchTrainer.biasVisible + biasVisUpdateMomentum))
    preDeltaUpdates.append((batchTrainer.biasHidden, batchTrainer.biasHidden + biasHidUpdateMomentum))

    updates = []

    if self.sparsityConstraint:
      if self.sparsityCostFunction == T.nnet.binary_crossentropy:
        sparistyCostMeasure = batchTrainer.activationProbabilities
      else:
        sparistyCostMeasure = batchTrainer.hiddenActivations

      runningAvg = batchTrainer.runningAvgExpected * 0.9 + T.mean(sparistyCostMeasure, axis=0) * 0.1
      # Sum over all hidden units
      sparsityCost = T.sum(self.sparsityCostFunction(self.sparsityTraget, runningAvg))

      updates.append((batchTrainer.runningAvgExpected, runningAvg))


    positiveDifference = T.dot(batchTrainer.visible.T, batchTrainer.hiddenActivations)
    negativeDifference = T.dot(batchTrainer.visibleReconstruction.T,
                               batchTrainer.hiddenReconstruction)
    delta = positiveDifference - negativeDifference

    # # Sparsity cost
    # if self.sparsityConstraint:
    #   gradientW = T.grad(sparsityCost, batchTrainer.weights)
    #   delta -= self.sparsityRegularization * gradientW

    if self.rmsprop:
      meanW = 0.9 * batchTrainer.oldMeanW + 0.1 * delta ** 2
      wUpdate = factorLr * batchLearningRate * delta / T.sqrt(meanW + 1e-8)
      updates.append((batchTrainer.oldMeanW, meanW))
    else:
      wUpdate = factorLr * batchLearningRate * delta

    wUpdate -= batchLearningRate * self.weightDecay * batchTrainer.oldDw

    updates.append((batchTrainer.weights, batchTrainer.weights + wUpdate))
    updates.append((batchTrainer.oldDw, wUpdate + wUpdateMomentum))

    visibleBiasDiff = T.sum(batchTrainer.visible - batchTrainer.visibleReconstruction, axis=0)


    if self.rmsprop:
      meanVis = 0.9 * batchTrainer.oldMeanVis + 0.1 * visibleBiasDiff ** 2
      biasVisUpdate = factorLr * batchLearningRate * visibleBiasDiff / T.sqrt(meanVis + 1e-8)
      updates.append((batchTrainer.oldMeanVis, meanVis))
    else:
      biasVisUpdate = factorLr * batchLearningRate * visibleBiasDiff

    updates.append((batchTrainer.biasVisible, batchTrainer.biasVisible + biasVisUpdate))
    updates.append((batchTrainer.oldDVis, biasVisUpdate + biasVisUpdateMomentum))

    hiddenBiasDiff = T.sum(batchTrainer.hiddenActivations - batchTrainer.hiddenReconstruction, axis=0)

    # As the paper says, only update the hidden bias
    if self.sparsityConstraint:
      gradientbiasHid = T.grad(sparsityCost, batchTrainer.biasHidden)
      hiddenBiasDiff -= self.sparsityRegularization * gradientbiasHid

    if self.rmsprop:
      meanHid = 0.9 * batchTrainer.oldMeanHid + 0.1 * hiddenBiasDiff ** 2
      biasHidUpdate = factorLr * batchLearningRate * hiddenBiasDiff / T.sqrt(meanHid + 1e-8)
      updates.append((batchTrainer.oldMeanHid, meanHid))
    else:
      biasHidUpdate = factorLr * batchLearningRate * hiddenBiasDiff

    updates.append((batchTrainer.biasHidden, batchTrainer.biasHidden + biasHidUpdate))
    updates.append((batchTrainer.oldDHid, biasHidUpdate + biasHidUpdateMomentum))

    # Add the updates required for the theano random generator
    updates += batchTrainer.updates.items()

    updates.append((batchTrainer.cdSteps, cdSteps))

    return preDeltaUpdates, updates


  #  Even though this function has no side effects, we need mini batches in
  # order to ensure that we do not go out of memory
  def hiddenRepresentation(self, dataInstances):
    dataInstacesConverted = theano.shared(np.asarray(dataInstances, dtype=theanoFloat))

    miniBatchSize = 1000
    nrMiniBatches = len(dataInstances) / nrMiniBatches + 1


    representHidden = theano.function(
            inputs=[index],
            outputs=self.reconstructer.hiddenActivations,
            updates=self.reconstructer.updates,
            givens={self.x: dataInstacesConverted[index * miniBatchSize: (index + 1) * miniBatchSize]})

    data = np.vstack([representHidden(miniBatchIndex) for i in xrange(nrMiniBatches)])

    return data

  #  Even though this function has no side effects, we need mini batches in
  # order to ensure that we do not go out of memory
  def reconstruct(self, dataInstances, cdSteps=1):
    dataInstacesConverted = theano.shared(np.asarray(dataInstances, dtype=theanoFloat))

    miniBatchSize = 1000
    nrMiniBatches = len(dataInstances) / nrMiniBatches + 1

    reconstructFunction = theano.function(
            inputs=[index],
            outputs=self.reconstructer.visibleReconstruction,
            updates=self.reconstructer.updates,
            givens={self.x: dataInstacesConverted[index * miniBatchSize: (index + 1) * miniBatchSize]})

    data = np.vstack([reconstructFunction(miniBatchIndex) for i in xrange(nrMiniBatches)])

    return data

  def reconstructionError(self, dataInstances):
    reconstructions = self.reconstruct(dataInstances)
    return rmse(reconstructions, dataInstances)

  def buildReconstructerForSymbolicVariable(self, x, theanoRng):
    reconstructer = ReconstructerBatch(input=x,
                                        theanoGenerator=theanoRng,
                                        weights=self.sharedWeights,
                                        biases=self.sharedBiases,
                                        visibleActivationFunction=self.visibleActivationFunction,
                                        hiddenActivationFunction=self.hiddenActivationFunction,
                                        visibleDropout=self.visibleDropout,
                                        hiddenDropout=self.hiddenDropout,
                                        cdSteps=1)
    return reconstructer


def initializeWeights(nrVisible, nrHidden):
  return  np.asarray(np.random.uniform(
                      low=-4 * np.sqrt(6. / (nrHidden + nrVisible)),
                      high=4 * np.sqrt(6. / (nrHidden + nrVisible)),
                      size=(nrVisible, nrHidden)), dtype=theanoFloat)
  # return np.asarray(np.random.normal(0, 0.01, (nrVisible, nrHidden)), dtype=theanoFloat)

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

def testWeights(weights, visibleDropout, hiddenDropout):
    return weights.T * hiddenDropout, weights * visibleDropout
