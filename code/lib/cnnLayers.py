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
class BatchTrainer(object):

  def __init__(self, layers, batchLearningRate):
    self.output = layers[-1].output
    self.batchLearningRate = batchLearningRate

    # Create the params of the trainer which will be used for gradient descent
    self.params = concatenateLists([l.params for l in layers])


  def cost(self, y):
    return T.nnet.categorical_crossentropy(self.output, y)


  # TODO: think if there is a simple way to reuse momentum, rmsprop and all that from
  # the deep belief net code
  # The easiest way to do this is move everything in a BatchTrainer class that is common
  def buildUpdates(self, error):
    deltas =  T.grad(error, self.params)

    updates = []

    for param, delta in zip(self.params, deltas):
      updates.append((param, param - self.batchLearningRate * delta))

    return updates


# TODO: move from here
class ConvolutionalNN(object):

  """

  """
  def __init__(self, layers, miniBatchSize, learningRate):
    self.layers = layers
    self.miniBatchSize = miniBatchSize
    self.learningRate = learningRate

  def _setUpLayers(self, x, inputDimensions):

    inputVar = x
    inputDimensionsPrevious = inputDimensions

    for layer in self.layers[0:-1]:
      layer._setUp(inputVar, inputDimensionsPrevious)
      inputDimensionsPrevious = layer._outputDimensions()
      inputVar = layer.output

    # the fully connected layer, the softmax layer
    # TODO: if you allow (and you should) multiple all to all layers you need to change this
    # after some point

    self.layers[-1]._setUp(inputVar.flatten(2),
                           inputDimensionsPrevious[0] * inputDimensionsPrevious[1] * inputDimensionsPrevious[2])


  def _reshapeInputData(self, data):
    if len(data[0].shape) == 2:
      inputShape = (data.shape[0], 1, data[0].shape[0], data[0].shape[1])
      data = data.reshape(inputShape)

    return data

  def train(self, data, labels, epochs=100):

    print "shuffling training data"
    data, labels = shuffle(data, labels)


    print "data.shape"
    print data.shape

    print "labels.shape"
    print labels.shape

    data = self._reshapeInputData(data)

    sharedData = theano.shared(np.asarray(data, dtype=theanoFloat))
    sharedLabels = theano.shared(np.asarray(labels, dtype=theanoFloat))

    nrMinibatches = len(data) / self.miniBatchSize


    batchLearningRate = self.learningRate / self.miniBatchSize
    batchLearningRate = np.float32(batchLearningRate)

    # Symbolic variable for the data matrix
    x = T.tensor4('x', dtype=theanoFloat)
    # the labels
    y = T.matrix('y', dtype=theanoFloat)

    # Set up the input variable as a field of the conv net
    # so that we can access it easily for testing
    self.x = x

    miniBatchIndex = T.lscalar()

    # Set up the layers with the appropriate theano structures
    self._setUpLayers(x, data[0].shape)

    #  create the batch trainer and using it create the updates
    batchTrainer = BatchTrainer(self.layers, batchLearningRate)

    # Set the batch trainer as a field in the conv net
    # then we can access it for a forward pass during testing
    self.batchTrainer = batchTrainer
    error = T.sum(batchTrainer.cost(y))
    updates = batchTrainer.buildUpdates(error)

    # the train function
    trainModel = theano.function(
            inputs=[miniBatchIndex],
            outputs=error,
            updates=updates,
            givens={
                x: sharedData[miniBatchIndex * self.miniBatchSize: (miniBatchIndex + 1) * self.miniBatchSize],
                y: sharedLabels[miniBatchIndex * self.miniBatchSize: (miniBatchIndex + 1) * self.miniBatchSize]})


    #  run the loop that trains the net
    for epoch in xrange(epochs):
      for i in xrange(nrMinibatches):
        trainModel(i)


  def test(self, data):
    miniBatchIndex = T.lscalar()

    data = self._reshapeInputData(data)
    sharedData = theano.shared(np.asarray(data, dtype=theanoFloat))

    # Do a forward pass trough the network
    forwardPass = theano.function(
            inputs=[miniBatchIndex],
            outputs=self.batchTrainer.output,
            givens={
                self.x: sharedData[miniBatchIndex * self.miniBatchSize: (miniBatchIndex + 1) * self.miniBatchSize]})

    nrMinibatches = data.shape[0] / self.miniBatchSize

    # do the loop that actually predicts the data
    lastLayer = concatenateLists([forwardPass(i) for i in xrange(nrMinibatches)])
    lastLayer = np.array(lastLayer)

    return lastLayer, np.argmax(lastLayer, axis=1)



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

  net = ConvolutionalNN(layers, 10, 0.1)

  trainData, trainLabels =\
      readmnist.read(0, 1000, digits=None, bTrain=True, path="../MNIST", returnImages=True)

  # transform the labels into vector (one hot encoding)
  trainLabels = labelsToVectors(trainLabels, 10)
  net.train(trainData, trainLabels, epochs=100)

  testData, testLabels =\
      readmnist.read(0, 100, digits=None, bTrain=False, path="../MNIST", returnImages=True)

  outputData, labels = net.test(testData)

  for i in xrange(100):
    print "labels", labels[i]
    print "testLabels", testLabels[i]

  print " "
  print "accuracy"
  print (labels == testLabels) / 100

if __name__ == '__main__':
  main()




