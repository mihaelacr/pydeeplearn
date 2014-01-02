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
def backprop(weights, topLayerDerivatives, y, z, derivativeFunctionVec):
  # vectorized derivative function
  derivativesForZ = derivativeFunctionVec(z) * topLayerDerivatives
  bottomLayerActivation = np.dot(weights, derivativesForZ)

  # Matrix, same shape as weights
  weightDerivatives = np.outer(y, derivativesForZ)
  assert weights.shape == weightDerivatives.shape

  return weightDerivatives, bottomLayerActivation

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

    # form now on assume discriminative training and then fix it for generative
    # if I actually implmenet wake sleep at some point
    nrRbms = nrLayers - 2

    # weights = []
    currentData = data
    for i in xrange(nrRbms):
      net = rbm()




def softmax(activation):
  expVec = np.vectorize(lambda x: math.exp(x), dtype=float)
  out = expVec(activation)
  return out / out.sum()
