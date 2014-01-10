import numpy as np

import restrictedBoltzmannMachine as rbm

# TODO: use conjugate gradient for  backpropagation instead of stepeest descent
# TODO: add weight decay in back prop
# TODO: implement dropout: it has been shown to improve learning
# TODO: monitor the changes in erorr and change the learning rate according
# to that
# TODO: wake sleep for improving generation

"""In all the above topLayer does not mean the top most layer, but rather the
layer above the current one."""

from common import *

""" Class that implements a deep blief network, for classifcation """
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
    discriminative: if the network is discriminative, then the last
        layer is required to be a softmax, in order to output the class
        probablities
  """
  def __init__(self, nrLayers, layerSizes, activationFunctions,
               discriminative=True):
    self.nrLayers = nrLayers
    self.layerSizes = layerSizes
    # Note that for the first one the activatiom function does not matter
    # So for that one there is no need to pass in an activation function
    self.activationFunctions = activationFunctions
    self.initialized = False
    self.discriminative = True

    # Simple checks
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
    # train the RBMS and set the weights
    # the weihghts can be stored as a list of numpy nd-arrays
    # if labels == None and self.discriminative == True:
    #   raise Exception("need labels for discriminative training")

    nrRbms = self.nrLayers - 1 - self.discriminative

    self.weights = []
    self.biases = []
    currentData = data
    for i in xrange(nrRbms):
      net = rbm.RBM(self.layerSizes[i], self.layerSizes[i+1],
                    rbm.contrastiveDivergence, self.activationFunctions[i].value)
      net.train(currentData)
      self.weights += [net.weights]
      self.biases += [net.biases[1]]

      currentData = net.hiddenRepresentation(currentData)

    # CHECK THAT
    self.weights += [np.random.normal(0, 0.01,
                                 (self.layerSizes[-2], self.layerSizes[-1]))]

    # Think of this
    self.biases += [np.random.normal(0, 0.01, self.layerSizes[-1])]

    assert len(self.weights) == self.nrLayers - 1
    assert len(self.biases) == self.nrLayers - 1
    # Does backprop or wake sleep?
    self.fineTune(data, labels)

  """Fine tunes the weigths and biases using backpropagation.

    Arguments:
      data: The data used for traning and fine tuning
      labels: A numpy nd array. Each label should be transformed into a binary
          base vector before passed into this function.
      miniBatch: The number of instances to be used in a miniBatch
      epochs: The number of epochs to use for fine tuning
  """
  # TODO: implement the minibatch business
  def fineTune(self, data, labels, miniBatchSize=8, epochs=100):
    learningRate = 0.01
    batchLearningRate = learningRate / miniBatchSize

    nrMiniBatches = len(data) / miniBatchSize

    oldDWeights = zerosFromShape(self.weights)
    oldDBias = zerosFromShape(self.biases)

    stages = len(self.weights)

    # TODO: maybe find a better way than this to find a stopping criteria
    for epoch in xrange(epochs):

      if epoch < 10:
        momentum = 0.5
      else:
        momentum = 0.95

      for batch in xrange(nrMiniBatches):
        start = batch * miniBatchSize
        end = (batch + 1) * miniBatchSize
        batchData = data[start: end]

        # this is a list of layer activities
        layerValues = self.forwardPass(batchData)
        finalLayerErrors = derivativesCrossEntropyError(labels[start:end],
                                              layerValues[-1])

        # Compute all derivatives
        dWeights, dBias = backprop(self.weights, layerValues,
                            finalLayerErrors, self.activationFunctions)

        # Update the weights and biases using gradient descent
        # Also update the old weights
        for index in xrange(stages):
          oldDWeights[index] = momentum * oldDWeights[index] + batchLearningRate * dWeights[index]
          oldDBias[index] = momentum * oldDBias[index] + batchLearningRate * dBias[index]
          self.weights[index] -= oldDWeights[index]
          self.biases[index] -= oldDBias[index]

  """Does a forward pass trought the network and computes the values of the
    neurons in all the layers.
    Required for backpropagation and classification.

    Arguments:
      dataInstaces: The instances to be run trough the network.
    """
  def forwardPass(self, dataInstaces):
    currentLayerValues = dataInstaces
    layerValues = [currentLayerValues]
    size = dataInstaces.shape[0]

    for stage in xrange(self.nrLayers - 1):
      weights = self.weights[stage]
      bias = self.biases[stage]
      activation = self.activationFunctions[stage]

      linearSum = np.dot(currentLayerValues, weights) + np.tile(bias, (size, 1))
      currentLayerValues = activation.value(linearSum)
      layerValues += [currentLayerValues]

    return layerValues

  def classify(self, dataInstaces):
    lastLayerValues = self.forwardPass(dataInstaces)[-1]
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
    upperLayerErrors = np.dot(deDz, weights[layer - 1].T)

    dw = np.einsum('ij,ik->jk', layerValues[layer - 1], deDz)

    dbias = deDz.sum(axis=0)

    # Iterating in decreasing order of layers, so we are required to
    # append the weight derivatives at the front as we go along
    deDw.insert(0, dw)
    deDbias.insert(0, dbias)

  return deDw, deDbias


""" Computes the derivatives of the top most layer given their output and the
target labels. This is computed using the cross entropy function.
See: http://en.wikipedia.org/wiki/Cross_entropy for the discrete case.
Since it is used with a softmax unit for classification, the output of the unit
represent a discrete probablity distribution and the expected values are
composed of a base vector, with 1 for the correct class and 0 for all the rest.
"""
def derivativesCrossEntropyError(expected, actual):
  # avoid dividing by 0 by adding a small number
  return - expected * (1.0 / actual)
