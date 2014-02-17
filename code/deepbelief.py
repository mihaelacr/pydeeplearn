import numpy as np

import restrictedBoltzmannMachine as rbm
import theano
from theano import tensor as T

theanoFloat  = theano.config.floatX

"""In all the above topLayer does not mean the top most layer, but rather the
layer above the current one."""

from common import *

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


  # the data and labels need to be theano stuff
  def train(self, data, labels=None):
    # This depends if you have generative or not
    nrRbms = self.nrLayers - 2

    self.weights = []
    self.biases = []
    currentData = data

    for i in xrange(nrRbms):
      net = rbm.RBM(self.layerSizes[i], self.layerSizes[i+1],
                    rbm.contrastiveDivergence,
                    1, 1,
                    self.activationFunctions[i].value)
      net.train(currentData)
      # you need to make the weights and biases shared and
      # add them to params
      w = theano.shared(value=np.asarray(net.weights / self.dropout,
                                         dtype=theanoFloat),
                        name='W')
      self.weights += [w]

      # Store the biases on GPU and do not return it on CPU (borrow=True)
      b = theano.shared(value=np.asarray(net.biases[1] / self.dropout,
                                         dtype=theanoFloat),
                        name='b')
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
    self.layerValues = []
    for i in xrange(self.nrLayers):
      vals = np.zeros(shape=(self.miniBatchSize, self.layerSizes[i]),
                                dtype=theanoFloat)
      layerVals = theano.shared(value=vals,
                      name='layerVals')
      self.layerValues.append(layerVals)

    # Does backprop or wake sleep?
    # Check if wake sleep works for real values
    self.fineTune(data, labels)
    # Change the weights according to dropout rules
    # make this shared maybe as well? so far they are definitely not
    # a problem so we will see later
    self.classifcationWeights = map(lambda x: x * self.dropout, self.weights)
    self.classifcationBiases = map(lambda x: x * self.dropout, self.biases)

  """Fine tunes the weigths and biases using backpropagation.
    Arguments:
      data: The data used for traning and fine tuning
        data has to be a theano variable for it to work in the current version
      labels: A numpy nd array. Each label should be transformed into a binary
          base vector before passed into this function.
      miniBatch: The number of instances to be used in a miniBatch
      epochs: The number of epochs to use for fine tuning
  """
  def fineTune(self, data, labels, epochs=100):
    # First step: let's make the data and the labels shared variables
    # so that we can nicely work with them

    # TODO: see if you have to use borrow here but probably not
    # because it only has effect on CPU
    sharedData = theano.shared(np.asarray(data,
                                               dtype=theano.config.floatX))
    # the cast might not be needed in my code because I do not think
    # I use the labels as indices, but I need to check this
    sharedLabels = T.cast(theano.shared(np.asarray(labels,
                                               dtype=theano.config.floatX)),
                          'int32')
    # def shared_dataset(data_xy, borrow=True):
    #     """ Function that loads the dataset into shared variables

    #     The reason we store our dataset in shared variables is to allow
    #     Theano to copy it into the GPU memory (when code is run on GPU).
    #     Since copying data into the GPU is slow, copying a minibatch everytime
    #     is needed (the default behaviour if the data is not in a shared
    #     variable) would lead to a large decrease in performance.
    #     """
    #     data_x, data_y = data_xy
    #     shared_x = theano.shared(numpy.asarray(data_x,
    #                                            dtype=theano.config.floatX),
    #                              borrow=borrow)
    #     shared_y = theano.shared(numpy.asarray(data_y,
    #                                            dtype=theano.config.floatX),
    #                              borrow=borrow)
    #     # When storing data on the GPU it has to be stored as floats
    #     # therefore we will store the labels as ``floatX`` as well
    #     # (``shared_y`` does exactly that). But during our computations
    #     # we need them as ints (we use labels as index, and if they are
    #     # floats it doesn't make sense) therefore instead of returning
    #     # ``shared_y`` we will have to cast it to int. This little hack
    #     # lets ous get around this issue
    #     return shared_x, T.cast(shared_y, 'int32')


    learningRate = 0.1
    batchLearningRate = learningRate / self.miniBatchSize

    nrMiniBatches = len(data) / self.miniBatchSize

    # oldDWeights = zerosFromShape(self.weights)
    # oldDBias = zerosFromShape(self.biases)

    stages = len(self.weights)

    # Let's build the symbolic graph which takes the data trough the network
    # allocate symbolic variables for the data
    # index of a mini-batch
    miniBatchIndex = T.lscalar()
    # The mini-batch data is a matrix
    x = T.matrix('x')
    # The labels, a vector
    y = T.ivector('y') # labels[start:end]

    error = self.cost(y)

    deltaParams = []
    # this is either a weight or a bias
    for param in self.params:
        delta = T.grad(error, param)
        deltaParams.append(delta)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    for param, delta in zip(self.params, deltaParams):
        updates.append((param, param - batchLearningRate * delta))

    train_model = theano.function(inputs=[index], outputs=error,
            updates=updates,
            givens={
                x: sharedData[index * self.miniBatchSize:(index + 1) * self.miniBatchSize],
                y: sharedLabels[index * self.miniBatchSize:(index + 1) * self.miniBatchSize]})

    # TODO: early stopping
    for epoch in xrange(epochs):
      # When you do early stopping you have to return the error on this batch
      # so that you can see when you stop or not
      for batchNr in xrange(nrMiniBatches):
        train_model(batchNr)

      # for all indices, call the functions defined above
      # Do not confuse calling an epoch to an index

      # for batch in xrange(nrMiniBatches):
      #   # for them the minibatch is also symbolic
      #   start = batch * self.miniBatchSize
      #   end = (batch + 1) * self.miniBatchSize
      #   batchData = data[start: end]

      #   # Do a forward pass with the batch data
      #   self.forwardPass(batchData)

      #   print self.layerValues[-1]
      #   print error
      #   print type(error)
      #   gparams = T.grad(self.params, error)
      #   dWeights, dBias = backprop(self.weights, layerValues,
      #                       finalLayerErrors, self.activationFunctions)

  def cost(self, y):
    return  T.nnet.categorical_crossentropy(self.layerValues[-1], y)

  def classify(self, dataInstaces):
    lastLayerValues = forwardPass(self.classifcationWeights,
                                  self.classifcationBiases,
                                  self.activationFunctions,
                                  dataInstaces)[-1]
    return lastLayerValues, np.argmax(lastLayerValues, axis=1)

  """ Does not do dropout. Used for classification. """
  def forwardPass(self, dataInstaces):
    currentLayerValues = dataInstaces
    self.layerValues[0] = currentLayerValues
    print type(self.layerValues[0])

    for stage in xrange(len(self.weights)):
      w = self.weights[stage]
      b = self.biases[stage]
      linearSum = T.dot(currentLayerValues, w) + b
      currentLayerValues = T.nnet.sigmoid(linearSum)
      # activation = self.activationFunctions[stage]
      # currentLayerValues = activation.value(linearSum)
      self.layerValues[stage + 1] = currentLayerValues
