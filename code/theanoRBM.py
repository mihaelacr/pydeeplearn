import numpy as np
from common import *

class RBM(object):

  def __init__(self, nrVisible, nrHidden, trainingFunction, dropout,
                visibleDropout, activationFun=sigmoid):
    # dropout = 1 means no dropout, keep all the weights
    self.dropout = dropout
    # dropout = 1 means no dropout, keep all the weights
    self.visibleDropout = visibleDropout
    self.nrHidden = nrHidden
    self.nrVisible = nrVisible
    self.trainingFunction = trainingFunction
    self.activationFun = activationFun

  def train(self, data):
    # If the network has not been initialized yet, do it now
    # Ie if this is the time it is traning batch of traning
    if not self.initialized:
      # TODO: make this shared
      self.weights = self.initializeWeights(self.nrVisible, self.nrHidden)
      self.biases = self.intializeBiases(data, self.nrHidden)

    # The training function needs to update the shared variables
    # weights and biases
    self.trainingFunction(data, self.activationFun, self.dropout, self.visibleDropout)

    self.testWeights = self.weights * self.dropout

    assert self.weights.shape == (self.nrVisible, self.nrHidden)
    assert self.biases[0].shape[0] == self.nrVisible
    assert self.biases[1].shape[0] == self.nrHidden

  """ Reconstructs the data given using this boltzmann machine."""
  def reconstruct(self, dataInstances):
    return reconstruct(self.biases, self.testWeights, dataInstances,
                       self.activationFun)

  def hiddenRepresentation(self, dataInstances):
    return updateLayer(Layer.HIDDEN, dataInstances, self.biases,
                       self.testWeights, self.activationFun, True)

