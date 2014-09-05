import numpy as np
import numpy.random as random

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from activationfunctions import *
from common import *


theanoFloat  = theano.config.floatX

# TODO: maybe set initialWeights to None so that you allow the user to initialize in the layer
# and not care so much for the process


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


  def _setUp(self, input):
    initialWeights = random.normal(loc=0.0, scale=0.1, size=(self.nrKernels, input.shape[0], self.kernelSize[0], self.kernelSize[1]))
    initialBiases = np.zeros(self.nrKernels)

    W = theano.shared(value=np.asarray(initialWeights,
                                         dtype=theanoFloat),
                        name='W')
    b = theano.shared(value=np.asarray(initialBiases,
                                         dtype=theanoFloat),
                        name='b')


    self.output = self.activationFun.deterministic(conv.conv2(input, W) + b.dimshuffle('x', 0, 'x', 'x'))

    self.params = [W, b]


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


  def _setUp(self, input):
    # downsample.max_pool_2d only downsamples on the last 2 dimensions of the input tensor
    self.output = downsample.max_pool_2d(input, self.poolingFactor, ignore_border=False)
    # each layer has to have a parameter field so that it is easier to concatenate all the parameters
    # when performing gradient descent
    self.params = []



class SoftmaxLayer(object):

  def __init__(self, size):
    self.size = size

  """
    input: 2D matrix
  """
  def _setUp(self, input):
    # can I get the size of the input even though it is a tensor var? should
    initialWeights = random.normal(loc=0.0, scale=0.1, size=(input.shape[1], self.size))
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
class BatchTrainer(object):

  def __init__(self, layers, batchLearningRate):
    self.output = layers[-1].output
    self.batchLearningRate = batchLearningRate

    # Create the params of the trainer which will be used for gradient descent
    self.params = list(itertools.chain.from_iterable([l.params for l in layers]))


  def cost(self, y):
    return T.nnet.categorical_crossentropy(self.output, y)


  # TODO: think if there is a simple way to reuse momentum, rmsprop and all that from
  # the deep belief net code
  # The easiest way to do this is move everything in a BatchTrainer class that is common
  def buildUpdates(self, error):
    deltas =  T.grad(error, self.params)

    updates = []

    for param, delta in zip(params, deltas):
      updates.append(param, param - self.batchLearningRate * delta)

    return updates


# TODO: move from here
class ConvolutionalNN(object):

  """
    layersSize
  """
  def __init__(self, layers, miniBatchSize, learningRate):
    self.layers = layers
    self.miniBatchSize = miniBatchSize
    self.learningRate = learningRate

  # TODO: not at all modular but let us see how this works
  def _setUpLayers(self, x):

    inputVar = x
    for i, layer in enumerate(layers):
      layer._setUp(inputVar)
      if i != len(layers) -1:
        inputVar = layer.output
      else:
        inputVar = layer.output.flatten(2)



  def train(self, data, labels, epochs=100):

    print "shuffling training data"
    data, labels = shuffle(data, labels)

    sharedData = theano.shared(np.asarray(data, dtype=theanoFloat))
    sharedLabels = theano.shared(np.asarray(labels, dtype=theanoFloat))

    nrMinibatches = len(data) / self.miniBatchSize


    batchLearningRate = self.learningRate / self.miniBatchSize
    batchLearningRate = np.float32(batchLearningRate)

    # Symbolic variable for the data matrix
    x = T.matrix('x', dtype=theanoFloat)

    # the labels
    y = T.matrix('y', dtype=theanoFloat)

    miniBatchIndex = T.lscalar()

    #  Create the layers and the mini batch trainer
    layers = self._setUpLayers(x)

    #  create the batch trainer and using it create the updates
    batchTrainer = BatchTrainer(layers)
    trainError = batchTrainer.cost(y)
    updates = batchTrainer.buildUpdates(error)

    # the train function
    trainModel = theano.function(
            inputs=[miniBatchIndex],
            outputs=trainError,
            updates=updates,
            givens={
                x: sharedData[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize],
                y: sharedLabels[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize]})


    #  run the loop that trains the net
    for epoch in xrange(epochs):
      for i in xrange(nrMinibatches):
        trainModel(i)


  def test(self, data):
    # the usual: do a forward pass and from that get what you need
    pass


# Let's build a simple convolutional neural network for classification
# Note that the last layer has to perform sub sampling such that you only
# end up with a vector

def main():
  # Import here because we need to make sure that this gets removed when refactoring
  import sys
  # We need this to import other modules
  sys.path.append("..")
  from read import readmnist

  layer1 = ConvolutionalLayer(50, (5, 5) , Sigmoid())
  layer2 = PoolingLayer((2, 2))
  layer3 = ConvolutionalLayer(20, (5, 5), Sigmoid())
  layer4 = PoolingLayer((2, 2))
  layer5 = SoftmaxLayer(10)

  layers = [layer1, layer2, layer3, layer4, layer5]

  net = ConvolutionalNN(layers, 100, 0.01)

  # start reading from the 55000 example because I do not want a lot of examples
  trainVectors, trainLabels =\
      readmnist.read(55000, args.trainSize, digits=None, bTrain=True, path=args.path)
  net.train(trainData, trainLabels)

if __name__ == '__main__':
  main()




