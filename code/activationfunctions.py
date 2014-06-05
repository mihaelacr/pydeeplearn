""" This class defines activation function that can be used with the nets in this project"""

from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import theano
import numpy as np


theanoFloat  = theano.config.floatX

class ActivationFunction(object):

  def __getstate__(self):
    odict = self.__dict__.copy() # copy the dict since we change it
    if 'theanoGenerator' in odict:
      del odict['theanoGenerator']
    return odict

  def __setstate__(self, dict):
    self.__dict__.update(dict)   # update attributes

  def __getinitargs__():
    return None


class Sigmoid(ActivationFunction):

  def __init__(self):
    self.theanoGenerator = RandomStreams(seed=np.random.randint(1, 1000))

  def nonDeterminstic(self, x):
    val = self.deterministic(x)
    return self.theanoGenerator.binomial(size=val.shape,
                                            n=1, p=val,
                                            dtype=theanoFloat)

  def deterministic(self, x):
    return T.nnet.sigmoid(x)


  def activationProbablity(self, x):
    return T.nnet.sigmoid(x)

class Rectified(ActivationFunction):

  def __init__(self):
    pass

  def nonDeterminstic(self, x):
    return self.deterministic(x)

  def deterministic(self, x):
    return x * (x > 0.0)

class RectifiedNoisy(ActivationFunction):

  def __init__(self):
    self.theanoGenerator = RandomStreams(seed=np.random.randint(1, 1000))

  def nonDeterminstic(self, x):
    x += self.theanoGenerator.normal(avg=0.0, std=(T.sqrt(T.nnet.sigmoid(x)) + 1e-08))
    return x * (x > 0.0)

  def deterministic(self, x):
    return expectedValueGaussian(x, T.nnet.sigmoid(x))

  def activationProbablity(self, x):
    return 1.0 - cdf(0, miu=x, variance=T.nnet.sigmoid(x))

class RectifiedNoisyVar1(ActivationFunction):

  def __init__(self):
    self.theanoGenerator = RandomStreams(seed=np.random.randint(1, 1000))

  def nonDeterminstic(self, x):
    x += self.theanoGenerator.normal(avg=0.0, std=1.0)
    return x * (x > 0.0)

  def deterministic(self, x):
    return expectedValueGaussian(x, 1.0)

  def activationProbablity(self, x):
    return 1.0 - cdf(0, miu=x, variance=1.0)

class Identity(ActivationFunction):

  def deterministic(self, x):
    return x

class Softmax(ActivationFunction):

  def deterministic(self, v):
    # Do not use theano's softmax, it is numerically unstable
    # and it causes Nans to appear
    # Semantically this is the same
    e_x = T.exp(v - v.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


# TODO: try this for the non deterministic version as well
class CappedRectifiedNoisy(ActivationFunction):

  def __init__(self):
    pass

  def nonDeterminstic(self, x):
    return self.deterministic(x)

  def deterministic(self, x):
    return x * (x > 0.0) * (x < 6.0)

# TODO
  def activationProbablity(self, x):
    return None


def expectedValueGaussian(mean, variance):
  std = T.sqrt(variance)
  return std / T.sqrt(2.0 * np.pi) * T.exp(- mean**2 / (2.0 * std)) + mean * cdf(mean / std)

# Approximation of the cdf of a standard normal
def cdf(x, miu=0.0, variance=1.0):
  return 1.0/2 *  (1.0 + T.erf((x - miu)/ T.sqrt(2 * variance)))

