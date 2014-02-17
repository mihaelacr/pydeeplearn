import numpy as np

import restrictedBoltzmannMachine as rbm
import theano
from theano import tensor as T

theanoFloat  = theano.config.floatX

"""In all the above topLayer does not mean the top most layer, but rather the
layer above the current one."""

from common import *

# This is a better logical unit
# than having the dbn store the layer values
# this is because the layer values

class MiniBatchTrainer(object):

  def __init__(self, input, nrLayers, initialWeights, initialBiases):
    self.input = input

    # Let's initialize the fields
    # The weights and biases, make them shared variables
    self.weights = []
    self.biases = []
    for i in xrange(nrLayers):
      w = theano.shared(value=np.asarray(initialWeights[i],
                                         dtype=theanoFloat),
                        name='W')
      self.weights.append(w)

      b = theano.shared(value=np.asarray(initialBiases[i],
                                         dtype=theanoFloat),
                        name='b')
      self.biases.append(b)

    # Is this needed?
    # self.layerValues = []
    # for i in xrange(self.nrLayers):
    #   vals = np.zeros(shape=(self.miniBatchSize, self.layerSizes[i]),
    #                             dtype=theanoFloat)
    #   layerVals = theano.shared(value=vals,
    #                   name='layerVals')
    #   self.layerValues.append(layerVals)
    currentLayerValues = self.inputs
    self.layerValues[0] = currentLayerValues
    for stage in xrange(len(self.weights)):
      w = self.weights[stage]
      b = self.biases[stage]
      linearSum = T.dot(currentLayerValues, w) + b
      # TODO: make this a function that you pass around
      currentLayerValues = T.nnet.sigmoid(linearSum)
      # activation = self.activationFunctions[stage]
      # currentLayerValues = activation.value(linearSum)
      self.layerValues[stage + 1] = currentLayerValues

    # Set the parameters of the object
    # Do not set more than this, these will be used for differentiation in the
    # gradient
    self.params = self.weights + self.biases


  def cost(self, y):
    return  T.nnet.categorical_crossentropy(self.layerValues[-1], y)

""" Class that implements a deep belief network, for classification """
class DBN(object):

  """
  Arguments:
    nrLayers: the number of layers of the network. In case of discriminative
        traning, also contains the classifcation layer
        (the last softmax layer)
        type: integer
    layerSizes: the sizes of the individual layers.
        type: list of integers of size nrLayers
    activationFunctions: the functions that are used to transform
        the input of a neuron into its output. The functions should be
        vectorized (as per numpy) to be able to apply them for an entire
        layer.
        type: list of objects of type ActivationFunction
  """
  def __init__(self, nrLayers, layerSizes, activationFunctions,
               dropout=0.5, rbmDropout=0.5, visibleDropout=0.8, rbmVisibleDropout=1):
    self.nrLayers = nrLayers
    self.layerSizes = layerSizes
    # Note that for the first one the activatiom function does not matter
    # So for that one there is no need to pass in an activation function
    self.activationFunctions = activationFunctions

    assert len(layerSizes) == nrLayers
    assert len(activationFunctions) == nrLayers - 1
    self.dropout = 1
    # you need a list of shared weights
    # the params are the params of the rbms + the softmax layer
    self.miniBatchSize = 10

  def train(self, data, labels=None):

    # This depends if you have generative or not
    nrRbms = self.nrLayers - 2

    self.weights = []
    self.biases = []

    # TODO: see if you have to use borrow here but probably not
    # because it only has effect on CPU
    sharedData = theano.shared(np.asarray(data,
                                               dtype=theano.config.floatX))
    # the cast might not be needed in my code because I do not think
    # I use the labels as indices, but I need to check this
    sharedLabels = T.cast(theano.shared(np.asarray(labels,
                                               dtype=theano.config.floatX)),
                          'int32')
    currentData = data

    for i in xrange(nrRbms):
      net = rbm.RBM(self.layerSizes[i], self.layerSizes[i+1],
                    rbm.contrastiveDivergence,
                    1, 1,
                    self.activationFunctions[i].value)
      net.train(currentData)
      # you need to make the weights and biases shared and
      # add them to params
      w = net.weights / self.dropout
      self.weights += [w]

      # Store the biases on GPU and do not return it on CPU (borrow=True)
      b = net.biases[1]
      self.biases += [b]

      currentData = net.hiddenRepresentation(currentData)

    # This depends if you have generative or not
    # Initialize the last layer of weights to zero if you have
    # a discriminative net
    lastLayerWeights = np.zeros(shape=(self.layerSizes[-2], self.layerSizes[-1]),
                                dtype=theanoFloat)

    w = theano.shared(value=lastLayerWeights,
                      name='W')
                      # borrow=True)

    lastLayerBiases = np.zeros(shape=(self.layerSizes[-1]),
                                dtype=theanoFloat)
    b = theano.shared(value=lastLayerBiases,
                      name='b')
                      # borrow=True)

    self.weights += [w]
    self.biases += [b]

    assert len(self.weights) == self.nrLayers - 1
    assert len(self.biases) == self.nrLayers - 1

    # Set the parameters of the net
    # According to them we will do backprop
    self.params = self.weights + self.biases

    # Create layervalues as shared variables (and symbolic automatically)

    # I have to set this input somehow and this is most likely to be done
    # with another class that has the batch stuff

    self.fineTune(sharedData, sharedLabels)
    # Change the weights according to dropout rules
    # make this shared maybe as well? so far they are definitely not
    # a problem so we will see later
    self.classifcationWeights = map(lambda x: x * self.dropout, self.weights)
    self.classifcationBiases = map(lambda x: x * self.dropout, self.biases)

  """Fine tunes the weigths and biases using backpropagation.
    data and labels are shared

    Arguments:
      data: The data used for traning and fine tuning
        data has to be a theano variable for it to work in the current version
      labels: A numpy nd array. Each label should be transformed into a binary
          base vector before passed into this function.
      miniBatch: The number of instances to be used in a miniBatch
      epochs: The number of epochs to use for fine tuning
  """
  def fineTune(self, data, labels, epochs=100):
    learningRate = 0.1
    batchLearningRate = learningRate / self.miniBatchSize

    nrMiniBatches = len(data) / self.miniBatchSize

    stages = len(self.weights)
    # Let's build the symbolic graph which takes the data trough the network
    # allocate symbolic variables for the data
    # index of a mini-batch
    miniBatchIndex = T.lscalar()
    # The mini-batch data is a matrix
    x = T.matrix('x')
    # The labels, a vector
    y = T.ivector('y') # labels[start:end]

    # here is where you can create the layered object
    # the mdb and with it you associate the cost function
    # and you update it's parameters
    batchTrainer = MiniBatchTrainer()

    # the error is the sum of the individual errors
    error = T.sum(self.cost(y))

    deltaParams = []
    # this is either a weight or a bias
    for param in batchTrainer.params:
        delta = T.grad(error, param)
        deltaParams.append(delta)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    for param, delta in zip(batchTrainer.params, deltaParams):
        updates.append((param, param - batchLearningRate * delta))

    train_model = theano.function(inputs=[index], outputs=error,
            updates=updates,
            givens={
                x: data[index * self.miniBatchSize:(index + 1) * self.miniBatchSize],
                y: labels[index * self.miniBatchSize:(index + 1) * self.miniBatchSize]})

    # TODO: early stopping
    for epoch in xrange(epochs):
      # When you do early stopping you have to return the error on this batch
      # so that you can see when you stop or not
      for batchNr in xrange(nrMiniBatches):
        train_model(batchNr)



  def classify(self, dataInstaces):
    lastLayerValues = forwardPass(self.classifcationWeights,
                                  self.classifcationBiases,
                                  self.activationFunctions,
                                  dataInstaces)[-1]
    return lastLayerValues, np.argmax(lastLayerValues, axis=1)
