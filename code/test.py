# this file is made to see how theano works and the speedup
# it gives you on GPU versus a normal implementation on CPU (ran on my computer)

import theano
import theano.tensor as T
from theano import function, shared
import numpy as np


import time
x = T.matrix('x', dtype=theano.config.floatX)
y = T.matrix('y', dtype=theano.config.floatX)

# z = x + y
# f = function([x, y], z)
# sc = shared(np.tile(0, 30, 30, dtype=theano.config.floatX), name='sc')
sc = shared(np.zeros((10, 10), dtype = theano.config.floatX), name='sc')

# sc = T.dot(x, y)
mydot = function( [x,y], updates=( (sc, T.dot(x,y)), ))

# a = np.random.random_integers(0, 100, (1000, 1000))
# b = np.random.random_integers(0, 100, (1000, 1000))

# We need to declare the variables shared to run on GPU
a = np.tile(40.0, (1000, 1000))
b = np.tile(23.0, (1000, 1000))
print "go"

before = time.time()
c = mydot(a,b)
print c.sum()
print time.time() - before

# print f(np.array([[2,3]]), np.array([[4,5]]))