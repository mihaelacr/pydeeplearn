""" Use this file to check the speed difference between a big matrix multiplication
performed on the GPU or on the CPU.
"""

import theano
import theano.tensor as T
from theano import function, shared
import numpy as np


import time
x = T.matrix('x', dtype=theano.config.floatX)
y = T.matrix('y', dtype=theano.config.floatX)

sc = shared(np.zeros((10, 10), dtype = theano.config.floatX), name='sc')

mydot = function( [x,y], updates=( (sc, T.dot(x,y)), ))

# We need to declare the variables shared to run on GPU
a = np.ones((20000, 20000), dtype = theano.config.floatX) * 40.0
b = np.ones((20000, 20000), dtype = theano.config.floatX) * 23.0
print "go"

before = time.time()
mydot(a,b)
print time.time() - before

print sc.get_value().sum()

