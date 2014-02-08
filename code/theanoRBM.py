import numpy as np
from common import *

# Import theano tensor
import theano.tensor as T

theanoFloat  = theano.config.floatX

class RBM(object):

  # We need to set the weights as optional parameters for dbns, in case we initialize the
  # weights to the transpose of the layer below
  def __init__(self, nrVisible, nrHidden, trainingFunction, dropout,
                visibleDropout, weights=None, activationFun=sigmoid):
    # dropout = 1 means no dropout, keep all the weights
    self.dropout = dropout
    # dropout = 1 means no dropout, keep all the weights
    self.visibleDropout = visibleDropout
    self.nrHidden = nrHidden
    self.nrVisible = nrVisible
    self.trainingFunction = trainingFunction
    self.activationFun = activationFun

    # self.random = random numpy thing

    # need to do something with the weights

  def train(self, data):
    # If the network has not been initialized yet, do it now
    # Ie if this is the time it is traning batch of traning
    if not self.initialized:
      weights = initializeWeights(self.nrVisible, self.nrHidden)
      # DO I need the named argument named=W?
      self.weights = theano.shared(value=weights)

      # this is a weird thing. Can you have a list with theano?
      biases = intializeBiases(data, self.nrHidden)
      self.biases = theano.shared(value=biases)

    # The training function needs to update the shared variables
    # weights and biases
    # these will need to have updates and things
    self.trainingFunction(data, self.activationFun, self.dropout, self.visibleDropout)

    testWeights = self.weights.get_value() * self.dropout
    self.testWeights = theano.shared(value=testWeights)

    # assert self.weights.shape == (self.nrVisible, self.nrHidden)
    # assert self.biases[0].shape[0] == self.nrVisible
    # assert self.biases[1].shape[0] == self.nrHidden


  def updateLayer(self, layer, otherLayerValues, binary=False):

    size = otherLayerValues.shape[0]

    if layer == Layer.VISIBLE:
      activation = T.dot(otherLayerValues, self.weights.T)
    else:
      activation = T.dot(otherLayerValues, self.weights)

    probs = self.activationFun(np.tile(self.biases[layer], (size, 1)) + activation)

    if binary:
      # Sample from the distributions
      return

    return probs



  # """ Reconstructs the data given using this boltzmann machine."""
  # def reconstruct(self, dataInstances):
  #   return reconstruct(self.biases, self.testWeights, dataInstances,
  #                      self.activationFun)

  # def hiddenRepresentation(self, dataInstances):
  #   return updateLayer(Layer.HIDDEN, dataInstances, self.biases,
  #                      self.testWeights, self.activationFun, True)



def initializeWeights(cls, nrVisible, nrHidden):
  return np.asarray(np.random.normal(0, 0.01, (nrVisible, nrHidden)),
                    dtype=theanoFloat)

def intializeBiases(cls, data, nrHidden):
  # get the procentage of data points that have the i'th unit on
  # and set the visible vias to log (p/(1-p))
  percentages = data.mean(axis=0, dtype='float')
  vectorized = np.vectorize(safeLogFraction, otypes=[np.float])
  visibleBiases = vectorized(percentages)

  visibleBiases = np.asarray(visibleBiases, dtype=theanoFloat)
  hiddenBiases = np.asarray(np.zeros(nrHidden), dtype=theanoFloat)
  return np.array([visibleBiases, hiddenBiases])