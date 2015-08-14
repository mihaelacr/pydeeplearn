""" A neural network trainer with various options: rmsprop, nerterov, momentum etc.
This trainer is used for multiple types of neural nets: dbns and cnns and it is designed
to be adaptable to other types of networks as well.
"""

__author__ = "Mihaela Rosca"
__contact__ = "mihaela.c.rosca@gmail.com"

import numpy as np
import theano
from theano import tensor as T

import debug

DEBUG = False

class BatchTrainer(object):
  """
    Abstract class used to define updates on the parameters of neural networks
    during training.

    Subclasses must call the constructor of this class in their constructors, and
    have to define their cost function using a method called 'cost'.

    Supports momentum updates and nesterov updates, both with rmsprop
    or without (see the TrainingOptions class for more details in the available
    training options. Also supports L1 and L2 weight decay.
  """

  def __init__(self, params, oldUpdates, oldMeanSquares, weights):
    self.params = params
    self.oldUpdates = oldUpdates
    self.oldMeanSquares = oldMeanSquares
    self.weights = weights if weights else []


  def makeTrainFunction(self, x, y, data, labels, trainingOptions):
    error = T.sum(self.cost(y))

    for w in self.weights:
      error += trainingOptions.weightDecayL1 * T.sum(abs(w))
      error += trainingOptions.weightDecayL2 * T.sum(w ** 2)

    miniBatchIndex = T.lscalar()
    momentum = T.fscalar()

    if DEBUG:
      mode = theano.compile.MonitorMode(post_func=debug.detect_nan).excluding(
                                        'local_elemwise_fusion', 'inplace')
    else:
      mode = None

    if trainingOptions.nesterov:
      preDeltaUpdates, updates = self.buildUpdatesNesterov(error, trainingOptions, momentum)
      updateParamsWithMomentum = theano.function(
          inputs=[momentum],
          outputs=[],
          updates=preDeltaUpdates,
          mode=mode)

      updateParamsWithGradient = theano.function(
          inputs =[miniBatchIndex, momentum],
          outputs=error,
          updates=updates,
          givens={
              x: data[miniBatchIndex * trainingOptions.miniBatchSize:(miniBatchIndex + 1) * trainingOptions.miniBatchSize],
              y: labels[miniBatchIndex * trainingOptions.miniBatchSize:(miniBatchIndex + 1) * trainingOptions.miniBatchSize]},
          mode=mode)

      def trainModel(miniBatchIndex, momentum):
        updateParamsWithMomentum(momentum)
        return updateParamsWithGradient(miniBatchIndex, momentum)

    else:
      updates = self.buildUpdatesSimpleMomentum(error, trainingOptions, momentum)
      trainModel = theano.function(
            inputs=[miniBatchIndex, momentum],
            outputs=error,
            updates=updates,
            mode=mode,
            givens={
                x: data[miniBatchIndex * trainingOptions.miniBatchSize:(miniBatchIndex + 1) * trainingOptions.miniBatchSize],
                y: labels[miniBatchIndex * trainingOptions.miniBatchSize:(miniBatchIndex + 1) * trainingOptions.miniBatchSize]})

    # returns the function that trains the model
    return trainModel

  def buildUpdatesNesterov(self, error, trainingOptions, momentum):
    if trainingOptions.momentumFactorForLearningRate:
      lrFactor =  np.float32(1.0) - momentum
    else:
      lrFactor = np.float32(1.0)

    preDeltaUpdates = []
    for param, oldUpdate in zip(self.params, self.oldUpdates):
      preDeltaUpdates.append((param, param + momentum * oldUpdate))

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    deltaParams = T.grad(error, self.params)
    updates = []
    parametersTuples = zip(self.params,
                           deltaParams,
                           self.oldUpdates,
                           self.oldMeanSquares)

    for param, delta, oldUpdate, oldMeanSquare in parametersTuples:
      if trainingOptions.rmsprop:
        meanSquare = 0.9 * oldMeanSquare + 0.1 * delta ** 2
        paramUpdate = - lrFactor * trainingOptions.batchLearningRate * delta / T.sqrt(meanSquare + 1e-8)
        updates.append((oldMeanSquare, meanSquare))
      else:
        paramUpdate = - lrFactor * trainingOptions.batchLearningRate * delta

      newParam = param + paramUpdate

      updates.append((param, newParam))
      updates.append((oldUpdate, momentum * oldUpdate + paramUpdate))

    return preDeltaUpdates, updates

  def buildUpdatesSimpleMomentum(self, error, trainingOptions, momentum):
    if trainingOptions.momentumFactorForLearningRate:
      lrFactor = np.float32(1.0) - momentum
    else:
      lrFactor = np.float32(1.0)

    deltaParams = T.grad(error, self.params)
    updates = []
    parametersTuples = zip(self.params,
                           deltaParams,
                           self.oldUpdates,
                           self.oldMeanSquares)

    for param, delta, oldUpdate, oldMeanSquare in parametersTuples:
      paramUpdate = momentum * oldUpdate
      if trainingOptions.rmsprop:
        meanSquare = 0.9 * oldMeanSquare + 0.1 * delta ** 2
        paramUpdate += - lrFactor * trainingOptions.batchLearningRate * delta / T.sqrt(meanSquare + 1e-8)
        updates.append((oldMeanSquare, meanSquare))
      else:
        paramUpdate += - lrFactor * trainingOptions.batchLearningRate * delta

      newParam = param + paramUpdate

      updates.append((param, newParam))
      updates.append((oldUpdate, paramUpdate))

    return updates