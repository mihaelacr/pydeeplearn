import numpy as np

import restrictedBoltzmannMachine as rbm

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
    self.initialized = False
    self.dropout = dropout
    self.rbmDropout = rbmDropout
    self.visibleDropout = visibleDropout
    self.rbmVisibleDropout = rbmVisibleDropout

    assert len(layerSizes) == nrLayers
    assert len(activationFunctions) == nrLayers - 1

    """
    TODO:
    If labels = None, only does the generative training
      with fine tuning for generation, not for discrimintaiton
      TODO: what happens if you do both? do the fine tuning for generation and
      then do backprop for discrimintaiton
    """

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
      self.weights += [net.weights / self.dropout]
      self.biases += [net.biases[1]]

      currentData = net.hiddenRepresentation(currentData)

    # This depends if you have generative or not
    # Initialize the last layer of weights to zero if you have
    # a discriminative net
    self.weights += [np.zeros((self.layerSizes[-2], self.layerSizes[-1]))]
    self.biases += [np.zeros(self.layerSizes[-1])]

    assert len(self.weights) == self.nrLayers - 1
    assert len(self.biases) == self.nrLayers - 1
    # Does backprop or wake sleep?
    self.fineTune(data, labels)
    self.classifcationWeights = map(lambda x: x * self.dropout, self.weights)
    self.classifcationBiases = self.biases

  """Fine tunes the weigths and biases using backpropagation.
    Arguments:
      data: The data used for traning and fine tuning
      labels: A numpy nd array. Each label should be transformed into a binary
          base vector before passed into this function.
      miniBatch: The number of instances to be used in a miniBatch
      epochs: The number of epochs to use for fine tuning
  """
  def fineTune(self, data, labels, miniBatchSize=10, epochs=100):
    learningRate = 0.1
    batchLearningRate = learningRate / miniBatchSize

    nrMiniBatches = len(data) / miniBatchSize

    oldDWeights = zerosFromShape(self.weights)
    oldDBias = zerosFromShape(self.biases)

    stages = len(self.weights)

    # TODO: maybe find a better way than this to find a stopping criteria
    for epoch in xrange(epochs):

      if epoch < epochs / 10:
        momentum = 0.5
      else:
        momentum = 0.95

      for batch in xrange(nrMiniBatches):
        start = batch * miniBatchSize
        end = (batch + 1) * miniBatchSize
        batchData = data[start: end]

        # this is a list of layer activities
        layerValues = forwardPassDropout(self.weights, self.biases,
                                        self.activationFunctions, batchData,
                                        self.dropout, self.visibleDropout)
        finalLayerErrors = derivativesCrossEntropyError(labels[start:end],
                                              layerValues[-1])

        # Compute all derivatives
        dWeights, dBias = backprop(self.weights, layerValues,
                            finalLayerErrors, self.activationFunctions)

        # Update the weights and biases using gradient descent
        # Also update the old weights
        for index in xrange(stages):
          oldDWeights[index] = momentum * oldDWeights[index] - batchLearningRate * dWeights[index]
          oldDBias[index] = momentum * oldDBias[index] - batchLearningRate * dBias[index]
          self.weights[index] += oldDWeights[index]
          self.biases[index] += oldDBias[index]


  def classify(self, dataInstaces):
    lastLayerValues = forwardPass(self.classifcationWeights,
                                  self.classifcationBiases,
                                  self.activationFunctions,
                                  dataInstaces)[-1]
    return lastLayerValues, np.argmax(lastLayerValues, axis=1)

"""
Arguments:
  weights: list of numpy nd-arrays
  layerValues: list of numpy arrays, each array representing the values of the
      neurons obtained during a forward pass of the network
  finalLayerErrors: errors on the final layer, they depend on the error function
      chosen. For softmax activation function on the last layer, use cross
      entropy as an error function.
"""
def backprop(weights, layerValues, finalLayerErrors, activationFunctions):
  nrLayers = len(weights) + 1
  deDw = []
  deDbias = []
  upperLayerErrors = finalLayerErrors

  for layer in xrange(nrLayers - 1, 0, -1):
    deDz = activationFunctions[layer - 1].derivativeForLinearSum(
                            upperLayerErrors, layerValues[layer])
    # upperLayerErrors = np.dot(deDz, weights[layer - 1].T)
    upperLayerErrors = np.tensordot(deDz, weights[layer - 1].T, [[deDz.ndim - 1], [weights[layer - 1].T.ndim -2]])

    dw = np.einsum('ij,ik->jk', layerValues[layer - 1], deDz)

    dbias = deDz.sum(axis=0)

    # Iterating in decreasing order of layers, so we are required to
    # append the weight derivatives at the front as we go along
    deDw.insert(0, dw)
    deDbias.insert(0, dbias)

  return deDw, deDbias

""" Does not do dropout. Used for classification. """
def forwardPass(weights, biases, activationFunctions, dataInstaces):
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


"""Does a forward pass trought the network and computes the values of the
    neurons in all the layers.
    Required for backpropagation and classification.

    Arguments:
      dataInstaces: The instances to be run trough the network.
    """
def forwardPassDropout(weights, biases, activationFunctions,
                       dataInstaces, dropout, visibleDropout):
  # dropout on the visible units
  # generally this is around 80%
  visibleOn = sample(visibleDropout, dataInstaces.shape)
  thinnedValues = dataInstaces * visibleOn
  layerValues = [thinnedValues]
  size = dataInstaces.shape[0]

  for stage in xrange(len(weights)):
    w = weights[stage]
    b = biases[stage]
    activation = activationFunctions[stage]

    linearSum = np.dot(thinnedValues, w) + np.tile(b, (size, 1))
    currentLayerValues = activation.value(linearSum)
    # this is the way to do it, because of how backprop works the wij
    # will cancel out if the unit on the layer is non active
    # de/ dw_i_j = de / d_z_j * d_z_j / d_w_i_j = de / d_z_j * y_i
    # so if we set a unit as non active here (and we have to because
    # of this exact same reason and of ow we backpropagate)
    if stage != len(weights) - 1:

      on = sample(dropout, currentLayerValues.shape)
      thinnedValues = on * currentLayerValues
      layerValues += [thinnedValues]
    else:
      layerValues += [currentLayerValues]

  return layerValues


""" Computes the derivatives of the top most layer given their output and the
target labels. This is computed using the cross entropy function.
See: http://en.wikipedia.org/wiki/Cross_entropy for the discrete case.
Since it is used with a softmax unit for classification, the output of the unit
represent a discrete probablity distribution and the expected values are
composed of a base vector, with 1 for the correct class and 0 for all the rest.
"""
def derivativesCrossEntropyError(expected, actual):
  return - expected * (1.0 / actual)

# Only works with binary units
def wakeSleep():
  pass
  # need to alternate between wake and sleep pahses
