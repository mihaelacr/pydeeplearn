import numpy as np

import restrictedBoltzmannMachine as rbm
import theano
from theano import tensor as T

theanoFloat  = theano.config.floatX

# TODO: use conjugate gradient for  backpropagation instead of steepest descent
# see here for a theano example http://deeplearning.net/tutorial/code/logistic_cg.py
# TODO: add weight decay in back prop but especially with the constraint
# on the weights
# TODO: monitor the changes in error and change the learning rate according
# to that
# TODO: wake sleep for improving generation
# TODO: nesterov method for momentum

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

    # Do I need to initialize this to scalars
    self.dropout = dropout
    self.rbmDropout = rbmDropout
    self.visibleDropout = visibleDropout
    self.rbmVisibleDropout = rbmVisibleDropout

    assert len(layerSizes) == nrLayers
    assert len(activationFunctions) == nrLayers - 1
    # you need a list of shared weights
    # the params are the params of the rbms + the softmax layer

    # This depends if you have generative or not
    nrRbms = self.nrLayers - 2
    self.miniBatchSize = 10


    """
    TODO:
    If labels = None, only does the generative training
      with fine tuning for generation, not for discrimintaiton
      TODO: what happens if you do both? do the fine tuning for generation and
      then do backprop for discrimintaiton
    """

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
                    self.rbmDropout,
                    self.rbmVisibleDropout,
                    self.activationFunctions[i].value)
      net.train(currentData)
      # you need to make the weights and biases shared and
      # add them to params
      w = theano.shared(value=np.asarray(net.weights / self.dropout,
                                         dtype=theanoFloat),
                        name='W')
      self.weights += [w]
      # this takes the biases from the hidden units
      # the biases for the visible units are discarded.

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
      labels: A numpy nd array. Each label should be transformed into a binary
          base vector before passed into this function.
      miniBatch: The number of instances to be used in a miniBatch
      epochs: The number of epochs to use for fine tuning
  """
  def fineTune(self, data, labels, epochs=100):
    learningRate = 0.1
    batchLearningRate = learningRate / self.miniBatchSize

    nrMiniBatches = len(data) / self.miniBatchSize

    # oldDWeights = zerosFromShape(self.weights)
    # oldDBias = zerosFromShape(self.biases)

    stages = len(self.weights)

    # TODO: maybe find a better way than this to find a stopping criteria
    # maybe do this entire loop on the GPU?
    for epoch in xrange(epochs):

      if epoch < epochs / 10:
        momentum = 0.5
      else:
        momentum = 0.95

      for batch in xrange(nrMiniBatches):
        start = batch * self.miniBatchSize
        end = (batch + 1) * self.miniBatchSize
        batchData = data[start: end]

        # Do a forward pass with the batch data
        self.forwardPass(batchData)

        # finalLayerErrors = derivativesCrossEntropyError(labels[start:end],
        #                                       layerValues[-1])

        # Compute all derivatives
        # In the new version you just need to do an update of the shared variables
        # but a clear way of seeing what are the parameters for everything would be good.
        # the weights can be kept as a list
        # TODO: this is a list, you need a function as an error
        # see:
        # http://deeplearning.net/tutorial/code/mlp.py
        # and
        # http://deeplearning.net/tutorial/code/logistic_sgd.py
        # maybe if we set layerValues[-1] as some parameter and then we
        # take labels[start end then it will work]
        # do the cost nicely
        print self.layerValues[-1]
        error = T.nnet.categorical_crossentropy(self.layerValues[-1], labels[start:end])
        print error
        print type(error)
        gparams = T.grad(self.params, error)
        dWeights, dBias = backprop(self.weights, layerValues,
                            finalLayerErrors, self.activationFunctions)

        # apply the updates somehow

        # Update the weights and biases using gradient descent
        # Also update the old weights
        # for index in xrange(stages):
        #   oldDWeights[index] = momentum * oldDWeights[index] + batchLearningRate * dWeights[index]
        #   oldDBias[index] = momentum * oldDBias[index] + batchLearningRate * dBias[index]
        #   self.weights[index] -= oldDWeights[index]
        #   self.biases[index] -= oldDBias[index]


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



# """Does a forward pass trought the network and computes the values of the
#     neurons in all the layers.
#     Required for backpropagation and classification.

#     Arguments:
#       dataInstaces: The instances to be run trough the network.
#     """
  # def forwardPassDropout(self, dataInstaces):
  #   # dropout on the visible units
  #   # generally this is around 80%
  #   visibleOn = sample(self.visibleDropout, dataInstaces.shape)
  #   thinnedValues = dataInstaces * visibleOn
  #   layerValues = [thinnedValues]

  #   for stage in xrange(len(self.weights)):
  #     w = self.weights[stage]
  #     b = self.biases[stage]
  #     activation = self.activationFunctions[stage]

  #     # for now use tensor.tile but it does not have a gradient so does not work
  #     # well with symblic differentiation
  #     # tile does not work like this?
  #     linearSum = np.dot(thinnedValues, w.get_value()) + b.get_value()
  #     # np.exp(2)
  #     currentLayerValues = activation.value(linearSum)
  #     # this is the way to do it, because of how backprop works the wij
  #     # will cancel out if the unit on the layer is non active
  #     # de/ dw_i_j = de / d_z_j * d_z_j / d_w_i_j = de / d_z_j * y_i
  #     # so if we set a unit as non active here (and we have to because
  #     # of this exact same reason and of ow we backpropagate)
  #     if stage != len(weights) - 1:
  #       on = sample(self.dropout, currentLayerValues.shape)
  #       thinnedValues = on * currentLayerValues
  #       layerValues += [thinnedValues]
  #     else:
  #       layerValues += [currentLayerValues]

  #   return layerValues


