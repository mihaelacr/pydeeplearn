""" Types of conv net layers that pydeeplearn supports now."""

__author__ = "Mihaela Rosca"
__contact__ = "mihaela.c.rosca@gmail.com"

import numpy as np
import numpy.random as random

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from activationfunctions import *
from common import *

theanoFloat  = theano.config.floatX

class ConvolutionalLayer(object):
  """
  The input has to be a 4D tensor:
   1) the size of a mini-batch (we do a forward pass for multiple images at a time)
   2) the number of input channels (or number of kernels for the previous layer)
   3) height
   4) width

  The weights are also a 4D tensor:
    1) Nr filters at next layer (chosen hyperparameter)
    2) Nr filters at previous layer (note that if that is the input layer the number
                                    of filters is given by the number of input channels)
    3) Height at next layer (given by the size of the filters)
    4) Width at the next layer

  InitialWeights should be created randomly or with RBM.
  Note that for now we assume that we construct all possible receptive fields for convolutions.
  """
  def __init__(self, nrKernels, kernelSize, activationFun):
    self.activationFun = activationFun
    self.kernelSize = kernelSize
    self.nrKernels = nrKernels


  # Can you not do the same thing as with the oldparam? eval or not?
  def _setUp(self, input, inputDimensions):
    self.inputDimensions = inputDimensions
    nrKernelsPrevious = inputDimensions[0]

    initialWeights = random.normal(loc=0.0, scale=0.1,
                                   size=(self.nrKernels, nrKernelsPrevious, self.kernelSize[0], self.kernelSize[1]))
    initialBiases = np.zeros(self.nrKernels)

    W = theano.shared(value=np.asarray(initialWeights,
                                         dtype=theanoFloat),
                        name='Wconv')
    b = theano.shared(value=np.asarray(initialBiases,
                                         dtype=theanoFloat),
                        name='bconv')

    self.output = self.activationFun.deterministic(conv.conv2d(input, W) + b.dimshuffle('x', 0, 'x', 'x'))
    self.weights = [W]
    self.params = [W, b]

  def _outputDimensions(self):
    a = self.inputDimensions[1]
    b = self.inputDimensions[2]
    return (self.nrKernels, a - self.kernelSize[0] + 1, b - self.kernelSize[1] + 1)


# Possible types for pooling
# l2 pooling
# max pooling
# average pooling
# average pooling weighted by the distance to the center
class PoolingLayer(object):

  # TODO: implement average pooling
  # TODO: support different pooling and subsampling factors
  """
  Input is again a 4D tensor just like in ConvolutionalLayer

  Note that if you combine the pooling and the convolutional operation then you
  can save a bit of time by not applying the activation function before the subsampling.
  You can still try and do that as an optimization even if you have 2 layers separated.

  poolingFactor needs to be a 2D tuple (eg: (2, 2))
  """

  def __init__(self, poolingFactor):
    self.poolingFactor = poolingFactor

  def _setUp(self, input, inputDimensions):
    # The pooling operation does not change the number of kernels
    self.inputDimensions = inputDimensions
    # downsample.max_pool_2d only downsamples on the last 2 dimensions of the input tensor
    self.output = downsample.max_pool_2d(input, self.poolingFactor, ignore_border=False)
    # each layer has to have a parameter field so that it is easier to concatenate all the parameters
    # when performing gradient descent
    self.params = []
    self.weights = []


  def _outputDimensions(self):
    a = self.inputDimensions[1]
    b = self.inputDimensions[2]
    return (self.inputDimensions[0], a / self.poolingFactor[0], b / self.poolingFactor[1])


class SoftmaxLayer(object):

  def __init__(self, size):
    self.size = size

  """
    input: 2D matrix
  """
  def _setUp(self, input, inputDimensions):
    # can I get the size of the input even though it is a tensor var? should
    initialWeights = random.normal(loc=0.0, scale=0.1, size=(inputDimensions, self.size))
    initialBiases = np.zeros(self.size)

    W = theano.shared(value=np.asarray(initialWeights, dtype=theanoFloat),
                        name='Wsoftmax')
    b = theano.shared(value=np.asarray(initialBiases, dtype=theanoFloat),
                        name='bsoftmax')

    softmax = Softmax()
    linearSum = T.dot(input, W) + b
    currentLayerValues = softmax.deterministic(linearSum)

    self.output = currentLayerValues
    self.weights = [W]
    self.params = [W, b]
