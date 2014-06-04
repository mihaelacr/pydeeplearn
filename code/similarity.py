""" This module implements the idea of finding out emotions similarities
by using the experiments similar to what Hinton describes in his NRelu paper."""

import restrictedBoltzmannMachine as rbm

import numpy as np
import theano
from theano import tensor as T

from common import *
from similarity_utils import *

theanoFloat  = theano.config.floatX


class Trainer(object):

  def __init__(self, input1, input2, net):

    self.w = theano.shared(value=np.float32(0))
    self.b = theano.shared(value=np.float32(0))
    self.net = net

    self.oldMeanSquarew = theano.shared(value=np.float32(0))
    self.oldMeanSquareb = theano.shared(value=np.float32(0))
    self.oldMeanSquareWeights = theano.shared(value=np.zeros(self.net.weights.shape , dtype=theanoFloat))
    self.oldMeanSquareBias = theano.shared(value=np.zeros(self.net.biases[1].shape , dtype=theanoFloat))

    self.oldDw = theano.shared(value=np.float32(0))
    self.oldDb = theano.shared(value=np.float32(0))
    self.oldDWeights = theano.shared(value=np.zeros(self.net.weights.shape , dtype=theanoFloat))
    self.oldDBias = theano.shared(value=np.zeros(self.net.biases[1].shape , dtype=theanoFloat))

    hiddenBias = net.sharedBiases[1]
    # Do I need to add all biases? Probably only the hidden ones
    self.params = [self.w, self.b, self.net.sharedWeights, hiddenBias]
    self.oldDParams = [self.oldDw, self.oldDb, self.oldDWeights, self.oldDBias]
    self.oldMeanSquares =  [self.oldMeanSquarew, self.oldMeanSquareb, self.oldMeanSquareWeights, self.oldMeanSquareBias]


    _, weightForHidden = rbm.testWeights(self.net.sharedWeights,
          visibleDropout=self.net.visibleDropout, hiddenDropout=self.net.hiddenDropout)

    hiddenActivations1 = net.hiddenActivationFunction.deterministic(T.dot(input1, weightForHidden) + hiddenBias)
    hiddenActivations2 = net.hiddenActivationFunction.deterministic(T.dot(input2, weightForHidden) + hiddenBias)

    # Use the cosine distance between the two vectors of of activations
    cos = cosineDistance(hiddenActivations1, hiddenActivations2)

    self.cos = cos
    prob = 1.0 /( 1.0 + T.exp(self.w * cos + self.b))

    self.output = prob


class SimilarityNet(object):

  # TODO: add sizes and activation functions here as well
  # plus rbm learning rates
  def __init__(self, learningRate, maxMomentum, rbmNrVis, rbmNrHid, rbmLearningRate,
                visibleActivationFunction, hiddenActivationFunction,
                rbmDropoutVis, rbmDropoutHid, rmsprop,trainingEpochsRBM,
                nesterovRbm,
                sparsityConstraint, sparsityRegularization, sparsityTraget):

    self.learningRate = np.float32(learningRate)
    self.rmsprop = rmsprop
    self.rbmNrVis = rbmNrVis
    self.maxMomentum = np.float32(maxMomentum)
    self.rbmNrHid = rbmNrHid
    self.rbmLearningRate = np.float32(rbmLearningRate)
    self.rbmDropoutHid = rbmDropoutHid
    self.rbmDropoutVis = rbmDropoutVis
    self.trainingEpochsRBM = trainingEpochsRBM
    self.visibleActivationFunction = visibleActivationFunction
    self.hiddenActivationFunction = hiddenActivationFunction
    self.nesterovRbm = nesterovRbm

    self.sparsityConstraint = sparsityConstraint
    self.sparsityRegularization = sparsityRegularization
    self.sparsityTraget = sparsityTraget

    if rmsprop:
      print "training similarity net with rmsprop"

  def _trainRBM(self, data1, data2):
    data = np.vstack([data1, data2])

    net = rbm.RBM(self.rbmNrVis, self.rbmNrHid, self.rbmLearningRate,
                    hiddenDropout=self.rbmDropoutHid,
                    visibleDropout=self.rbmDropoutVis,
                    visibleActivationFunction=self.visibleActivationFunction,
                    hiddenActivationFunction=self.hiddenActivationFunction,
                    rmsprop=True,
                    nesterov=self.nesterovRbm,
                    trainingEpochs=self.trainingEpochsRBM,
                    sparsityConstraint=self.sparsityConstraint,
                    sparsityRegularization=self.sparsityRegularization,
                    sparsityTraget=self.sparsityTraget)
    net.train(data)

    return net


  def train(self, data1, data2, similarities, miniBatchSize=20, epochs=200):
    nrMiniBatches = len(data1) / miniBatchSize
    miniBatchIndex = T.lscalar()
    momentum = T.fscalar()
    learningRate = T.fscalar()

    learningRateMiniBatch = np.float32(self.learningRate / miniBatchSize)
    print "learningRateMiniBatch in similarity net"
    print learningRateMiniBatch

    net = self._trainRBM(data1, data2)

    data1  = theano.shared(np.asarray(data1,dtype=theanoFloat))
    data2  = theano.shared(np.asarray(data2,dtype=theanoFloat))
    similarities = theano.shared(np.asarray(similarities,dtype=theanoFloat))

    # The mini-batch data is a matrix
    x = T.matrix('x', dtype=theanoFloat)
    y = T.matrix('y', dtype=theanoFloat)
    self.x = x
    self.y = y

    z = T.vector('z', dtype=theanoFloat)

    trainer = Trainer(x, y, net)
    self.trainer = trainer

    # error = T.sum(T.sqr(trainer.output-z))
    error = T.sum(T.nnet.binary_crossentropy(trainer.output, z))

    updates = self.buildUpdates(trainer, error, learningRate, momentum)

    # Now you have to define the theano function
    discriminativeTraining = theano.function(
      inputs=[miniBatchIndex, learningRate, momentum],
      outputs=[trainer.output, trainer.cos],
      updates=updates,
      givens={
            x: data1[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize],
            y: data2[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize],
            z: similarities[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize],
            })

    for epoch in xrange(epochs):
      print "epoch", epoch
      momentum = np.float32(min(np.float32(0.5) + epoch * np.float32(0.1),
                       np.float32(0.95)))

      for miniBatch in xrange(nrMiniBatches):
        output, cos = discriminativeTraining(miniBatch, learningRateMiniBatch, momentum)

    print trainer.w.get_value()
    print trainer.b.get_value()

  def test(self, testData1, testData2):
    # If it is too slow try adding mini batches
    testData1 = np.array(testData1, dtype=theanoFloat)
    testData2 = np.array(testData2, dtype=theanoFloat)

    # TODO : think of making data1 and data2 shared
    testFunction = theano.function(
      inputs=[],
      outputs=self.trainer.output,
      givens={self.x: testData1,
            self.y: testData2
            })

    return testFunction()

  def buildUpdates(self, trainer, error, learningRate, momentum):
    if self.rmsprop:
      return self.buildUpdatesRmsprop(trainer, error, learningRate, momentum)
    else:
      return self.buildUpdatesNoRmsprop(trainer, error, learningRate, momentum)

  def buildUpdatesNoRmsprop(self, trainer, error, learningRate, momentum):
    updates = []
    gradients = T.grad(error, trainer.params)

    for param, oldParamUpdate, gradient in zip(trainer.params, trainer.oldDParams, gradients):
      paramUpdate = momentum * oldParamUpdate - learningRate * gradient
      updates.append((param, param + paramUpdate))
      updates.append((oldParamUpdate, paramUpdate))

    return updates

  def buildUpdatesRmsprop(self, trainer, error, learningRate, momentum):
    updates = []
    gradients = T.grad(error, trainer.params)

    for param, oldParamUpdate, oldMeanSquare, gradient in zip(trainer.params, trainer.oldDParams, trainer.oldMeanSquares, gradients):
      meanSquare = 0.9 * oldMeanSquare + 0.1 * gradient ** 2
      paramUpdate = momentum * oldParamUpdate - learningRate * gradient / T.sqrt(meanSquare + 1e-08)
      updates.append((param, param + paramUpdate))
      updates.append((oldParamUpdate, paramUpdate))
      updates.append((oldMeanSquare, meanSquare))

    return updates


def cosineDistance(first, second):
  normFirst = T.sqrt(T.sum(T.sqr(first), axis=1))
  normSecond = T.sqrt(T.sum(T.sqr(second), axis=1))
  return 1.0 - T.sum(first * second, axis=1) / (normFirst * normSecond)

