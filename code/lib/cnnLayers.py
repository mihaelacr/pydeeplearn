import numpy as np
import numpy.random as random

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from activationfunctions import *
from common import *


theanoFloat  = theano.config.floatX



# maybe set a field in each layer "isFullyConnected" that can tell you what you need to do depending
# on that: much more scalable and supports a bigger variety of architectures

#  Note that these layers that I have now are not transparent to the user in their creation
#  you can use the builder pattern to make them transparent and add the theano
# elements after the user has told you what they want
#  but in that case you do not get much advantage in between that and a string.

# you can have a basic init and a fill method that takes the current arguments for init

# if you think about it is is weird that the user has to deal with the input matrices and so on when they do not
# care about these things

# you could just add a 'setInput' method

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


  def _setUp(self, input, inputDimensions):
    self.inputDimensions = inputDimensions
    nrKernelsPrevious = inputDimensions[0]

    initialWeights = random.normal(loc=0.0, scale=0.1,
                                   size=(self.nrKernels, nrKernelsPrevious, self.kernelSize[0], self.kernelSize[1]))
    initialBiases = np.zeros(self.nrKernels)

    W = theano.shared(value=np.asarray(initialWeights,
                                         dtype=theanoFloat),
                        name='W')
    b = theano.shared(value=np.asarray(initialBiases,
                                         dtype=theanoFloat),
                        name='b')


    self.output = self.activationFun.deterministic(conv.conv2d(input, W) + b.dimshuffle('x', 0, 'x', 'x'))

    self.params = [W, b]

  def _outputDimensions(self):
    a = self.inputDimensions[1]
    b = self.inputDimensions[2]
    return (self.nrKernels, a - self.kernelSize[0] + 1, b - self.kernelSize[1] + 1)



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
                        name='W')
    b = theano.shared(value=np.asarray(initialBiases, dtype=theanoFloat),
                        name='b')

    softmax = Softmax()
    linearSum = T.dot(input, W) + b
    currentLayerValues = softmax.deterministic(linearSum)

    self.output = currentLayerValues

    self.params = [W, b]


# you could add the methods to a batch trainer class that could be common between
# cnns and dbns with the common feature trains
# TODO: you could even put the y in the constructor here
# TODO: move this from here and make the minibatch trainer from db
# to implement this
class BatchTrainer(object):

  def buildUpdates(self, error, batchLearningRate, momentum, nesterov,
                            momentumFactorForLearningRate, rmsprop):
    # deltas =  T.grad(error, self.params)
    # updates = []

    # for param, delta in zip(self.params, deltas):
    #   updates.append((param, param - batchLearningRate * delta))

    # return updates
    print "nesterov", nesterov
    if nesterov == True:
      return self.buildUpdatesNesterov(error, batchLearningRate, momentum,
                            momentumFactorForLearningRate, rmsprop)
    else:
      return self.buildUpdatesSimpleMomentum(error, batchLearningRate, momentum,
                                      momentumFactorForLearningRate, rmsprop)


  def buildUpdatesNesterov(self, error, batchLearningRate, momentum,
                            momentumFactorForLearningRate, rmsprop):

    if momentumFactorForLearningRate:
      lrFactor = 1.0 - momentum
    else:
      lrFactor = 1.0

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
                           self.oldMeanSquare)

    for param, delta, oldUpdate, oldMeanSquare in parametersTuples:
      if rmsprop:
        meanSquare = 0.9 * oldMeanSquare + 0.1 * delta ** 2
        paramUpdate = - lrFactor * batchLearningRate * delta / T.sqrt(meanSquare + 1e-8)
        updates.append((oldMeanSquare, meanSquare))
      else:
        paramUpdate = - lrFactor * batchLearningRate * delta

      newParam = param + paramUpdate

      updates.append((param, newParam))
      updates.append((oldUpdate, momentum * oldUpdate + paramUpdate))

    return preDeltaUpdates, updates

  def buildUpdatesSimpleMomentum(self, error, batchLearningRate, momentum,
                                 momentumFactorForLearningRate, rmsprop):
    if momentumFactorForLearningRate:
      lrFactor = 1.0 - momentum
    else:
      lrFactor = 1.0

    deltaParams = T.grad(error, self.params)
    updates = []
    parametersTuples = zip(self.params,
                           deltaParams,
                           self.oldUpdates,
                           self.oldMeanSquares)

    for param, delta, oldUpdate, oldMeanSquare in parametersTuples:
      paramUpdate = momentum * oldUpdate
      if rmsprop:
        meanSquare = 0.9 * oldMeanSquare + 0.1 * delta ** 2
        paramUpdate += - lrFactor * batchLearningRate * delta / T.sqrt(meanSquare + 1e-8)
        updates.append((oldMeanSquare, meanSquare))
      else:
        paramUpdate += - lrFactor * batchLearningRate * delta

      newParam = param + paramUpdate

      updates.append((param, newParam))
      updates.append((oldUpdate, paramUpdate))

    return updates



class CNNBatchTrainer(BatchTrainer):

  def __init__(self, layers):
    self.output = layers[-1].output

    # Create the params of the trainer which will be used for gradient descent
    self.params = concatenateLists([l.params for l in layers])

    # ok so now we define the old values using the eval function from theano
    # if this is too expensive we will just keep some fields in
    self.oldUpdates = []
    self.oldMeanSquares = []
    for param in self.params:
      oldDParam = theano.shared(value=np.zeros(shape=param.shape.eval(),
                                              dtype=theanoFloat),
                                name='oldDParam')

      self.oldUpdates += [oldDParam]
      oldMeanSquare = theano.shared(value=np.zeros(shape=param.shape.eval(),
                                              dtype=theanoFloat),
                                name='oldMeanSquare')

      self.oldMeanSquares += [oldMeanSquare]


  def cost(self, y):
    return T.nnet.categorical_crossentropy(self.output, y)
