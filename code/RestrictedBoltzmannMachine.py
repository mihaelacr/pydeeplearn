"""Implementation of restricted boltzmann machine

You need to be able to deal with different energy functions

This allows you to deal with real valued unit

do updates in parallel using multiprocessing.pool

TODO: monitor overfitting
TODO: weight decay (to control overfitting and other things)
TODO: mean filed and dumped mean field

"""
import numpy as np
import math

# Makes a step in the contrastiveDivergence algorithm
# online or with mini-bathces?
# you have multiple choices about how to implement this
def contrastiveDivergence(data, biases, weights, cdSteps=1):
  pass

# Another training algorithm. Slower than Contrastive divergence, but
# gives better results. Not used in practice as it is too slow.
def PCD():
  pass

"""
 Represents a RBM
"""
class RBM(object):
  # 2 layers
  # weights
  # biases for each layer.
  # you can have a type of RBM that just extends the kind of functions
  # you use to update, depending on what kind of energy you use, which also
  # decides what kind of units you have
  # data is a numpy array:
  def __init__(self, data, nrHidden, trainingFunction):
    # Initialize weights to random
    assert len(data) !=0
    self.nrHidden = nrHidden
    self.nrVisible = len(data[0])
    self.weights = initializeWeights(nrVisible, nrHidden)
    self.biases = intializeBiases(data, nrHidden)

  # you need to take a training algorithm as a parameter (CD, PCD)
  def train():
    self.weights = trainingFunction(self.data, self.biases, self.weights)

  @classmethod
  def initializeWeights(nrVisible, nrHidden):
    return np.random.normal(0, 0.01, (nrVisible, nrHidden))

  @classmethod
  def initalizeBiases(data, nrHidden):
    # get the procentage of data points that have the i'th unit on
    # and set the visible vias to log (p/(1-p))
    percentages = data.mean(axis=0, dtype='float') / len(data)
    # TODO:what happens if one of them is 1?
    vectorized = np.vectorize(lambda p: math.log(p / (1 -p)))
    visibleBiases = vectorized(percentages)

    # TODO: if sparse hiddeen weights, use that information
    hiddenBiases = np.zeros(nrHidden)
    return visibleBiases, hiddenBiases


""" general unitily functions"""

def sigmoid(x):
  return 1 / 1 + np.exp(-x);

def sample(p):
  if np.random.uniform() < p:
    return 1
  return 0

def sampleAll(probs):
  return map(sample, probs)