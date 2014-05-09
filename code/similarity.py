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

    self.oldDw = theano.shared(value=np.float32(0))
    self.oldDb = theano.shared(value=np.float32(0))
    self.oldDWeights = theano.shared(value=np.zeros(self.net.weights.shape , dtype=theanoFloat))
    self.oldDBias = theano.shared(value=np.zeros(self.net.biases[1].shape , dtype=theanoFloat))


    hiddenBias = net.sharedBiases[1]
    # Do I need to add all biases? Probably only the hidden ones
    self.params = [self.w, self.b, self.net.sharedWeights, hiddenBias]
    self.oldDParams = [self.oldDw, self.oldDb, self.oldDWeights, self.oldDBias]


    _, weightForHidden = rbm.testWeights(self.net.sharedWeights,
          visibleDropout=self.net.visibleDropout, hiddenDropout=self.net.hiddenDropout)

    hiddenActivations1 = T.nnet.sigmoid(T.dot(input1, weightForHidden) + hiddenBias)
    hiddenActivations2 = T.nnet.sigmoid(T.dot(input2, weightForHidden) + hiddenBias)

    # Here i have no sampling
    cos = cosineDistance(hiddenActivations1, hiddenActivations2)

    prob = 1.0 /( 1.0 + T.exp(self.w * cos + self.b))

    self.output = prob


class SimilarityNet(object):

  # TODO: add sizes and activation functions here as well
  # plus rbm learning rates
  def __init__(self, learningRate, maxMomentum, rbmNrVis, rbmNrHid, rbmLearningRate,
                rbmDropoutVis, rbmDropoutHid, binary):
    self.learningRate = learningRate
    self.binary = binary
    self.rbmNrVis = rbmNrVis
    self.maxMomentum = maxMomentum
    self.rbmNrHid = rbmNrHid
    self.rbmLearningRate = rbmLearningRate
    self.rbmDropoutHid = rbmDropoutHid
    self.rbmDropoutVis = rbmDropoutVis


  def _trainRBM(self, data1, data2):
    data = np.vstack([data1, data2])
    # TODO: activation function change
    activationFunction = T.nnet.sigmoid

    net = rbm.RBM(self.rbmNrVis, self.rbmNrHid, self.rbmLearningRate,
                    hiddenDropout=self.rbmDropoutHid,
                    visibleDropout=self.rbmDropoutVis,
                    binary=self.binary,
                    visibleActivationFunction=activationFunction,
                    hiddenActivationFunction=activationFunction,
                    rmsprop=True,
                    nesterov=True)
    net.train(data)

    return net


  def train(self, data1, data2, similarities, miniBatchSize=10, epochs=100):
    nrMiniBatches = len(data1) / miniBatchSize
    miniBatchIndex = T.lscalar()
    momentum = T.fscalar()


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

    error = T.sum(T.sqr(trainer.output-z))

    updates = self.buildUpdates(trainer, error, momentum)

    # Now you have to define the theano function
    discriminativeTraining = theano.function(
      inputs=[miniBatchIndex, momentum],
      outputs=[trainer.output],
      updates=updates,
      givens={
            x: data1[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize],
            y: data2[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize],
            z: similarities[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize],
            })

    for epoch in xrange(epochs):
      momentum = np.float32(min(np.float32(0.5) + epoch * np.float32(0.1),
                       np.float32(0.95)))

      for miniBatch in xrange(nrMiniBatches):
        discriminativeTraining(miniBatch, momentum)

    print trainer.w.get_value()
    print trainer.b.get_value()

  def test(self, testData1, testData2):
    # If it is too slow try adding mini batches

    testData1 = np.array(testData1, dtype=theanoFloat)
    testData2 = np.array(testData2, dtype=theanoFloat)

    # TODO : think of making data1 and data2 shared
    testFunction = theano.function(
      inputs=[],
      outputs=[self.trainer.output],
      givens={self.x: testData1,
            self.y: testData2
            })

    return testFunction()

  # You can add momentum and all that here as well
  def buildUpdates(self, trainer, error, momentum):
    updates = []
    gradients = T.grad(error, trainer.params)
    for param, oldParamUpdate, gradient in zip(trainer.params, trainer.oldDParams, gradients):
      paramUpdate = momentum * oldParamUpdate - self.learningRate * gradient
      updates.append((param, param + paramUpdate))
      updates.append((oldParamUpdate, paramUpdate))

    return updates


def cosineDistance(first, second):
  normFirst = T.sum(T.sqrt(first), axis=1)
  normSecond = T.sum(T.sqrt(second), axis=1)
  return T.sum(first * second, axis=1) / (normFirst * normSecond)

# Here  you need different measures than 0, 1 according to what you want it to learn
# for the emotions part
def defineSimilartyMesures():
  None

