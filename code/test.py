# this file is made to see how theano works and the speedup
# it gives you on GPU versus a normal implementation on CPU (ran on my computer)

import theano.tensor as T
from theano import function
import numpy as np


import time
x = T.matrix('x', dtype='int64')
y = T.matrix('y', dtype='int64')
# z = x + y
# f = function([x, y], z)


sc = T.dot(x, y)
mydot = function([x,y], sc)

# a = np.random.random_integers(0, 100, (1000, 1000))
# b = np.random.random_integers(0, 100, (1000, 1000))

a = np.tile(40, (1000, 1000))
b = np.tile(23, (1000, 1000))
print "go"

before = time.time()

c = mydot(a,b)
print time.time() - before

print c

# print f(np.array([[2,3]]), np.array([[4,5]]))