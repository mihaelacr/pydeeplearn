import numpy as np

import restrictedBoltzmannMachine as rbm
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

theanoFloat  = theano.config.floatX

"""In all the above topLayer does not mean the top most layer, but rather the
layer above the current one."""

from common import *


def detect_nan(i, node, fn):
    for output in fn.outputs:
        if np.isnan(output[0]).any():
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break

def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print "output(s) value(s):", [output[0] for output in fn.outputs]

class MiniBatchTrainer(object):

  # TODO: maybe creating the ring here might be better?
  def __init__(self, input, nrLayers, initialWeights, initialBiases,
               visibleDropout, hiddenDropout):
    self.input = input

    # Let's initialize the fields
    # The weights and biases, make them shared variables
    self.weights = []
    self.biases = []
    for i in xrange(nrLayers - 1):
      w = theano.shared(value=np.asarray(initialWeights[i],
                                         dtype=theanoFloat),
                        name='W')
      self.weights.append(w)

      b = theano.shared(value=np.asarray(initialBiases[i],
                                         dtype=theanoFloat),
                        name='b')
      self.biases.append(b)

    # Set the parameters of the object
    # Do not set more than this, these will be used for differentiation in the
    # gradient
    self.params = self.weights + self.biases

    # The updates that were performed in the last batch
    # Required for momentum
    # It is important that the order in which
    # we add the oldUpdates is the same as which we add the params
    # TODO: add an assertion for this
    self.oldUpdates = []
    for i in xrange(nrLayers - 1):
      oldDw = theano.shared(value=np.zeros(shape=initialWeights[i].shape,
                                           dtype=theanoFloat),
                        name='oldDw')
      self.oldUpdates.append(oldDw)

    for i in xrange(nrLayers - 1):
      oldDb = theano.shared(value=np.zeros(shape=initialBiases[i].shape,
                                           dtype=theanoFloat),
                        name='oldDb')
      self.oldUpdates.append(oldDb)


    # Create a theano random number generator
    # Required to sample units for dropout
    # If it is not shared, does it update when we do the
    # when we go to another function call?
    self.theano_rng = RandomStreams(seed=np.random.randint(1, 1000))
    # Note: do the optimization when you keep all of them:
    # this is required for classification

    # Sample from the visible layer
    # Get the mask that is used for the visible units
    dropout_mask = self.theano_rng.binomial(n=1, p=visibleDropout,
                                            size=self.input.shape,
                                            dtype=theanoFloat)
    currentLayerValues = self.input * dropout_mask

    # TODO: remove this list: it is unneeded
    # Just keep the output of the last layer
    self.layerValues = [0 for x in xrange(nrLayers)]
    self.layerValues[0] = currentLayerValues

    for stage in xrange(len(self.weights)):
      # Dropout: randomly select some weights to keep
      # Get the mask that is used to select which of
      # these hidden units we keep
      w = self.weights[stage]
      b = self.biases[stage]
      linearSum = T.dot(currentLayerValues, w) + b
      # TODO: make this a function that you pass around
      # it is important to make the classification activation functions outside
      # Also check the Stamford paper again to what they did to average out
      # the results with softmax and regression layers?
      if stage != len(self.weights) -1:
        dropout_mask = self.theano_rng.binomial(n=1, p=hiddenDropout,
                                            size=self.input.shape,
                                            dtype=theanoFloat)
        currentLayerValues = dropout_mask * T.nnet.sigmoid(linearSum)
      else:
        e_x = T.exp(linearSum - linearSum.max(axis=1, keepdims=True))
        currentLayerValues = e_x / e_x.sum(axis=1, keepdims=True)
        # currentLayerValues = T.nnet.softmax(linearSum)

      self.layerValues[stage + 1] = currentLayerValues

  def cost(self, y):
    return T.nnet.categorical_crossentropy(self.layerValues[-1], y)

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
    self.miniBatchSize = 10

  def train(self, data, labels=None):

    # This depends if you have generative or not
    nrRbms = self.nrLayers - 2

    self.weights = []
    self.biases = []

    sharedData = theano.shared(np.asarray(data, dtype=theanoFloat))
    sharedLabels = theano.shared(np.asarray(labels, dtype=theanoFloat))

    # Train the restricted Boltzmann machines that form the network
    currentData = data
    for i in xrange(nrRbms):
      net = rbm.RBM(self.layerSizes[i], self.layerSizes[i+1],
                    rbm.contrastiveDivergence,
                    1, 1,
                    self.activationFunctions[i].value)
      net.train(currentData)

      w = net.weights
      self.weights += [w]
      b = net.biases[1]
      self.biases += [b]

      # Let's update the current representation given to the next RBM
      currentData = net.hiddenRepresentation(currentData)

    # This depends if you have generative or not
    # Initialize the last layer of weights to zero if you have
    # a discriminative net
    lastLayerWeights = np.zeros(shape=(self.layerSizes[-2], self.layerSizes[-1]),
                                dtype=theanoFloat)
    lastLayerBiases = np.zeros(shape=(self.layerSizes[-1]),
                               dtype=theanoFloat)

    self.weights += [lastLayerWeights]
    self.biases += [lastLayerBiases]

    assert len(self.weights) == self.nrLayers - 1
    assert len(self.biases) == self.nrLayers - 1

    self.nrMiniBatches = len(data) / self.miniBatchSize

    # Does backprop for the data and a the end sets the weights
    self.fineTune(sharedData, sharedLabels)

    # Dropout: Get the classification
    self.classifcationWeights = map(lambda x: x * self.dropout, self.weights)
    self.classifcationBiases = self.biases

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
    batchLearningRate = np.float32(batchLearningRate)

    nrMiniBatches = self.nrMiniBatches
    # Let's build the symbolic graph which takes the data trough the network
    # allocate symbolic variables for the data
    # index of a mini-batch
    miniBatchIndex = T.lscalar()
    momentum = T.fscalar()

    # The mini-batch data is a matrix
    x = T.matrix('x', dtype=theanoFloat)
    # The labels, a vector
    y = T.matrix('y', dtype=theanoFloat) # labels[start:end] this needs to be a matrix because we output probabilities



    # Check if this actually updates the weights
    # of the DBN o it does it at the end
    batchTrainer = MiniBatchTrainer(input=x, nrLayers=self.nrLayers,
                                    initialWeights=self.weights,
                                    initialBiases=self.biases,
                                    visibleDropout=0.8,
                                    hiddenDropout=0.5)

    # the error is the sum of the errors in the individual cases
    error = T.sum(batchTrainer.cost(y))
    deltaParams = T.grad(error, batchTrainer.params)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    parametersTuples = zip(batchTrainer.params, deltaParams, batchTrainer.oldUpdates)
    for param, delta, oldUpdate in parametersTuples:
        paramUpdate = momentum * oldUpdate - batchLearningRate * delta
        newParam = param + paramUpdate
        updates.append((param, newParam))
        updates.append((oldUpdate, paramUpdate))

    mode = theano.compile.MonitorMode(
      post_func=detect_nan).excluding(
    'local_elemwise_fusion', 'inplace')

    train_model = theano.function(
            inputs=[miniBatchIndex, momentum],
            outputs=error,
            updates=updates,
            givens={
                x: data[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize],
                y: labels[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize]})

    # TODO: early stopping
    # TODO: do this loop in THEANO to increase speed?
    for epoch in xrange(epochs):
      print "in if"

      for batchNr in xrange(nrMiniBatches):
        if epoch < epochs / 10:
          momentum = np.float32(0.5)
        else:
          momentum = np.float32(0.95)
        error = train_model(batchNr, momentum)

    # Set up the weights in the dbn object
    for i in xrange(len(self.weights)):
      self.weights[i] = batchTrainer.weights[i].get_value()

    for i in xrange(len(self.biases)):
      self.biases[i] = batchTrainer.biases[i].get_value()


  def classify(self, dataInstaces):
    # TODO: run it on the gpu according to the number of instances
    # I think it is better to just run it on GPU
    # The mini-batch data is a matrix
    dataInstacesConverted = np.asarray(dataInstaces, dtype=theanoFloat)

    x = T.matrix('x', dtype=theanoFloat)

    # Use the classification weights because now we have dropout
    # Ensure that you have no dropout in classification
    # TODO: are the variables still shared? or can we make a new one?
    batchTrainer = MiniBatchTrainer(input=x, nrLayers=self.nrLayers,
                                    initialWeights=self.classifcationWeights,
                                    initialBiases=self.classifcationBiases,
                                    visibleDropout=1,
                                    hiddenDropout=1)

    classify = theano.function(
            inputs=[],
            outputs=batchTrainer.layerValues[-1],
            updates={},
            givens={x: dataInstacesConverted})

    lastLayers = classify()

    lastLayerValues = lastLayers

    # lastLayerValues = forwardPass(self.classifcationWeights,
    #                               self.classifcationBiases,
    #                               self.activationFunctions,
    #                               dataInstaces)[-1]
    return lastLayerValues, np.argmax(lastLayerValues, axis=1)


# NO LONGER REALLY USED? REMOVE?

# This method is now kept only for classification
# The training is done using theano and does not need this
# I will see if I add this later to GPU as well
# Since we now have the derivative there
# is not need to have the more convoluted classes for activation functions
# TODO: some of these things here are 64 bits, this can cause problems
def forwardPass(weights, biases, activationFunctions, dataInstaces):
  # TODO: data instances should be float32
  currentLayerValues = dataInstaces
  layerValues = [currentLayerValues]
  size = dataInstaces.shape[0]

  for stage in xrange(len(weights)):
    w = weights[stage]
    b = biases[stage]
    activation = activationFunctions[stage]

    linearSum = np.dot(currentLayerValues, w) + np.tile(b, (size, 1))
    currentLayerValues = activation.value(linearSum)
    layerValues += [currentLayerValues]

  return layerValues
