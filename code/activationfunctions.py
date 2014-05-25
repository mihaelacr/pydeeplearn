""" This class defines activation function that can be used with the nets in this project"""

from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import theano
import numpy as np


theanoFloat  = theano.config.floatX

class Sigmoid(object):

  def __init__(self):
    self.theanoGenerator = RandomStreams(seed=np.random.randint(1, 1000))

  def nonDeterminstic(self, x):
    val = self.deterministc(x)
    return self.theanoGenerator.binomial(size=val.shape,
                                            n=1, p=val,
                                            dtype=theanoFloat)

  def deterministc(self, x):
    return T.nnet.sigmoid(x)

class Rectified(object):

  def __init__(self):
    pass

  def nonDeterminstic(self, x):
    return self.deterministc(x)

  def deterministc(self, x):
    return x * (x > 0.0)

class RectifiedNoisy(object):

  def __init__(self):
    self.theanoGenerator = RandomStreams(seed=np.random.randint(1, 1000))

  def nonDeterminstic(self, x):
    x += self.theanoGenerator.normal(avg=0.0, std=T.nnet.ultra_fast_sigmoid(x))
    return x * (x > 0.0)

  def deterministc(self, x):
    return expectedValueGaussian(x, T.nnet.ultra_fast_sigmoid(x))

class RectifiedNoisyVar1(object):

  def __init__(self):
    self.theanoGenerator = RandomStreams(seed=np.random.randint(1, 1000))

  def nonDeterminstic(self, x):
    x += self.theanoGenerator.normal(avg=0.0, std=1.0)
    return x * (x > 0.0)

  def deterministc(self, x):
    return expectedValueGaussian(x, 1.0)


def expectedValueGaussian(mean, std):
  return T.sqrt(std / (2.0 * np.pi)) * T.exp(- mean**2 / (2.0 * std)) + mean * cdf(mean / std)

# Approximation of the cdf of a standard normal
def cdf(x):
  return 1.0/2 *  (1.0 + T.erf(x / T.sqrt(2)))


def identity(var):
  return var

# Do not use theano's softmax, it is numerically unstable
# and it causes Nans to appear
# Semantically this is the same
def softmax(v):
  e_x = T.exp(v - v.max(axis=1, keepdims=True))
  return e_x / e_x.sum(axis=1, keepdims=True)
