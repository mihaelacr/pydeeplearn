

import theano
from theano import tensor as T
from theano.tensor.nnet import conv



theanoFloat  = theano.config.floatX


class ConvolutionalLayer(object):

  """
  The input has to be a 4D tensor:
   1) the size of a mini-batch (we do a forward pass for multiple images at a time)
   2) the number of input channels (or number of kernels for the previous layer)
   3) height
   4) width

  The weights are also a 4D tensor:
    1) Nr filters at next layer (chosen hyperparameter)
    2) Nr filters at previous layer (note that if that is the input layer the number
                                    of filters is given by the number of input channels)
    3) Height at next layer (given by the size of the filters)
    4) Width at the next layer

  InitialWeights should be created randomly or with RBM.
  Note that for now we assume that we construct all possible receptive fields for convolutions.
  """
  def __init__(self, input, initialWeights, initialBiases, activationFun):

    self.W = theano.shared(value=np.asarray(initialWeights,
                                         dtype=theanoFloat),
                        name='W')
    self.b = theano.shared(value=np.asarray(initialBiases,
                                         dtype=theanoFloat),
                        name='b')


    self.output = activationFun.deterministic(conv.conv2(input, self.W) + self.b.dimshuffle('x', 0, 'x', 'x'))

    self.params = [self.W, self.b]



class PoolingLayer(object):

  # TODO: implement also average pooling
  def __init__(self, poolingFactor, poolingFunction='max'):
    pass



