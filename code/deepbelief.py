import numpy as np

import restrictedBoltzmannMachine as rbm

# TODO: use momentum for backpropagation
# TODO: try tanh instead of the usual 1 /(1 + exp.(-x)) ?
# Note that this requires also changes in output function
# This function is also mentioned in bishop

# TODO: use conjugate gradient for  backpropagation instead of stepeest descent

"""In all the above topLayer does not mean the uppo """

from common import *


""" Class that implements a deep blief network, for classifcation """

class DBN(object):

  """
    Arguments:
      nrLayers: the number of layers of the network. In case of discriminative
        traning, also contains the classifcation layer (ie the last softmax layer)
        type: integer
      layerSizes: the sizes of the individual layers.
        type: list of integers of size nrLayers
      activationFunctions: the functions that are used to transform
        the input of a neuron into its output. The functions should be
        vectorized (as per numpy) to be able to apply them for an entire
        layer.
        type: list of objects of type ActivationFunction
      discriminative: if the network is discriminative, then the last
        layer is required to be a softmax, in order to output the class probablities
  """
  def __init__(self, nrLayers, layerSizes, activationFunctions, discriminative=True):
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
      TODO: what happens if you do both? do the fine tuning for generation and then
      do backprop for discrimintaiton
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
      net = rbm.RBM(self.layerSizes[i], self.layerSizes[i+1], rbm.contrastiveDivergence)
      net.train(currentData)
      self.weights += [net.weights]
      self.biases += [net.biases[1]]

      currentData = net.reconstruct(currentData)

    # The last softmax unit also has weights and biases, but it;s not a RBM
    # CHECK THAT
    self.weights += [np.random.normal(0, 0.01, (self.layerSizes[-2], self.layerSizes[-1]))]

    # Think of this
    self.biases += [np.random.normal(0, 0.01, self.layerSizes[-1])]

    assert len(self.weights) == self.nrLayers - 1
    # Does backprop or wake sleep?
    self.fineTune(data, labels)

  """Fine tunes the weigths and biases using backpropagation.

    Arguments:
      labels: A matrix, not a vector. Each label should be transformed into a binary b
        base vector before passed into this function.
  """
  # TODO: actually fine tune the biases as well.
  # TODO: implement the minibatch business
  def fineTune(self, data, labels, miniBatch=1, epochs=100):
    learningRate = 0.0001

    # TODO: maybe find a better way than this to find a stopping criteria
    for epoch in xrange(epochs):

      for i, d in enumerate(data):
        # this is a list of layer activities
        layerValues = self.forwardPass(d)

        finalLayerErrors = outputDerivativesCrossEntropyErrorFunction(labels[i], layerValues[-1])
        # Compute all derivatives
        dWeights = backprop(self.weights, layerValues, finalLayerErrors, self.activationFunctions)
        for w, dw in zip(self.weights, dWeights):
          w = w - learningRate * dw


  """Does a forward pass trought the network and computes the values of all the layers.
     Required for backpropagation and classification. """
  # TODO: think if you can do it with matrix stuff
  def forwardPass(self, dataInstace):
    currentLayerValues = dataInstace
    layerValues = [currentLayerValues]

    for stage in xrange(self.nrLayers - 1):
      weights = self.weights[stage]
      biases = self.biases[stage]
      activation = self.activationFunctions[stage]

      currentLayerValues = activation.value(np.dot(currentLayerValues, weights) + biases)
      layerValues += [currentLayerValues]

    return layerValues

  # Do not support this if not discriminative, but I think that makes no sense to implement
  # to be honest
  # implementing wake and sleep and backprop could be something
  # Do wake and sleep first nd then backprop: improve weights for generation
  # and then improve them for classification
  # TODO: get more data instances
  def classify(self, dataInstace):
    lastLayerValues = self.forwardPass(dataInstace)[-1]
    return lastLayerValues, indexOfMin(lastLayerValues)


"""
Arguments:
  weights: list of numpy nd-arrays
  layerValues: list of numpy nd-arrays
  finalLayerErrors: errors on the final layer, they depend on the error function chosen
"""
def backprop(weights, layerValues, finalLayerErrors, activationFunctions):
  # Compute the last layer derivatives for the softmax

  # assert deDz.shape == layerValues[-1].shape

  nrLayers = len(weights) + 1
  deDw = []

  upperLayerErrors = finalLayerErrors

  for layer in xrange(nrLayers - 1, 0, -1):
    deDz = activationFunctions[layer - 1].derivativeForLinearSum(upperLayerErrors, layerValues[layer])
    dw, dbottom = derivativesForBottomLayer(weights[layer - 1], layerValues[layer - 1], deDz)
    upperLayerErrors = dbottom

    # Iterating in decreasing order of layers, so we are required to
    # append the weight derivatives at the front as we go along
    deDw.insert(0, dw)

  assert len(deDw) == len(weights)

  return deDw


# Could make small clases that jus tapply the function and also ge tthe derivatives for it
def sigmoidDerivativeForLinearSum(topLayerDerivatives, topLayerActivations):
  return topLayerActivations * (1 - topLayerActivations) * topLayerDerivatives

""" Computes the derivatives of the top most layer given their output and the
target labels. This is computed using the cross entropy function.
See: http://en.wikipedia.org/wiki/Cross_entropy for the discrete case.
Since it is used with a softmax unit for classification, the output of the unit
represent a discrete probablity distribution and the expected values are
composed of a base vector, with 1 for the correct class and 0 for all the rest.
"""
def outputDerivativesCrossEntropyErrorFunction(expected, actual):
  return - expected * (1.0 / actual)

def softmaxDerivativeForLinearSum(topLayerDerivatives, topLayerActivations):
  # write it as matrix multiplication
  d = - np.outer(topLayerActivations, topLayerActivations)
  d[np.diag_indices_from(d)] = topLayerActivations * (1 - topLayerActivations)
  return np.dot(topLayerDerivatives, d)


"""
Arguments:
  weights: the weight matrix between the layers for which the derivatives are computed
rename y
"""
def derivativesForBottomLayer(layerWeights, y, derivativesWrtLinearInputSum):
  # vectorized derivative function
  # IMPORTANT: this will not work as gor sigmoid you put y in
  # does not work for softmax?
  # maybe compute the derivatives for z in a different function and pass it here
  bottomLayerDerivatives = np.dot(layerWeights, derivativesWrtLinearInputSum)

  # Matrix, same shape as layerWeights
  weightDerivatives = np.outer(y, derivativesWrtLinearInputSum)
  assert layerWeights.shape == weightDerivatives.shape

  return weightDerivatives, bottomLayerDerivatives
