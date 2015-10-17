__author__ = 'snurkabill'

from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import theano
import numpy as np

theanoFloat = theano.config.floatX

class CostFunction(object):

  def __getstate__(self):
    odict = self.__dict__.copy()
    if 'theanoGenerator' in odict:
      del odict['theanoGenerator']
    return odict

  def __setstate__(self, dict):
    self.__dict__.update(dict)

  def __getinitargs__():
    return None

class LeastSquares(CostFunction):

  def __init__(self):
    pass

  def cost(self, x, y):
    return (x - y) * (x - y)

  def __call__(self, *args, **kwargs):
      return self.cost(args[1], args[2])

class CategoricalCrossEntropy(CostFunction):

  def __init__(self):
    pass

  def __call__(self, *args, **kwargs):
      return self.cost(args[1], args[2])

  def cost(self, x, y):
    return T.nnet.categorical_crossentropy(x, y)
