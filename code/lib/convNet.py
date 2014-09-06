import numpy as np

import theano
from theano import tensor as T

from common import *
from cnnLayers import *

theanoFloat  = theano.config.floatX

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
    batchTrainer = CNNBatchTrainer(self.layers)

    # Set the batch trainer as a field in the conv net
    # then we can access it for a forward pass during testing
    self.batchTrainer = batchTrainer
    error = T.sum(batchTrainer.cost(y))
    updates = batchTrainer.buildUpdates(error, batchLearningRate, 1.0, False, False, False)

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
      readmnist.read(0, 100, digits=None, bTrain=True, path="../MNIST", returnImages=True)

  # transform the labels into vector (one hot encoding)
  trainLabels = labelsToVectors(trainLabels, 10)
  net.train(trainData, trainLabels, epochs=100)

  testData, testLabels =\
      readmnist.read(0, 10, digits=None, bTrain=False, path="../MNIST", returnImages=True)

  outputData, labels = net.test(testData)

  for i in xrange(10):
    print "labels", labels[i]
    print "testLabels", testLabels[i]

  print " "
  print "accuracy"
  print sum(labels == testLabels) * 1.0 / 10

if __name__ == '__main__':
  main()