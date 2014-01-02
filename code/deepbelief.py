import numpy as np


import restrictedBoltzmannMachine as rbm

from common import *

# Returns a vector of derivatives
"""
Arguments:
  weights: the weight matrix between the layers
rename y z
they have been computed already for the forward pass so no need to compute everything again
"""
def backprop(weights, y, derivativesWrtLinearInputSum):
  # vectorized derivative function
  # IMPORTANT: this will not work as gor sigmoid you put y in
  # does not work for softmax?
  # maybe compute the derivatives for z in a different function and pass it here
  bottomLayerDerivatives = np.dot(weights, derivativesWrtLinearInputSum)

  # Matrix, same shape as weights
  weightDerivatives = np.outer(y, derivativesForZ)
  assert weights.shape == weightDerivatives.shape

  return weightDerivatives, bottomLayerDerivatives

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
        type: list of functions
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
    if labels == None and self.discriminative == True:
      raise Exception("need labels for discriminative training")

    nrRbms = nrLayers - 1 - self.discriminative

    self.weights = []
    self.biases = []
    currentData = data
    for i in xrange(nrRbms):
      net = rbm(self.layerSizes[i], self.layerSizes[i+1], rbm.contrastiveDivergence)
      net.train(currentData)
      self.weights += net.weights
      self.biases += self.biases

      currentData = net.reconstruct(currentData)

    # Does backprop or wake sleep?
    self.fineTune(data)

  """Fine tunes the weigths and biases using backpropagation. """
  # TODO: actually fine tune the biases as well.
  def fineTune(self, data):
    # Define error function. Maybe a better error function than mean square error?

     for d in data:
      layerValues = forwardPass(d)

    pass

  """Does a forward pass trought the network and computes the values of all the layers.
     Required for backpropagation. """
  def forwardPass(self, dataInstace):
    # You have to use the layer's activation function
    currentLayerValues = dataInstace
    layerValues = []
    layerValues += [currentLayerValues]
    for stage in xrange(self.nrLayers - 1):
      weights = self.weights[stage]
      biases = self.biases[stage]
      fun = self.activationFunctions[stage]

      currentLayerValues = fun(np.dot(currentLayerValues, weights) + biases)
      layerValues += [currentLayerValues]

    return layerValues


def softmax(activation):
  expVec = np.vectorize(lambda x: math.exp(x), dtype=float)
  out = expVec(activation)
  return out / out.sum()

def sigmoidDerivativeForLinearSum(topLayerDerivatives, topLayerActivations):
  return topLayerActivations * (1 - topLayerActivations) * topLayerDerivatives


def softmaxDerivativeForLinearSum(topLayerDerivatives, topLayerActivations):
  return 1