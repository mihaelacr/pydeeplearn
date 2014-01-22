# this file is made to see how theano works and the speedup
# it gives you on GPU versus a normal implementation on CPU (ran on my computer)

import theano.tensor as T
from theano import function
import numpy as np

x = T.matrix('x')
y = T.matrix('y')
# z = x + y
# f = function([x, y], z)


sc = T.dot(x, y)
mydot = function([x,y], sc)

a = np.random.random_integers(0, 100, (10000, 10000))
b = np.random.random_integers(0, 100, (10000, 10000))
print mydot(a,b)
# print f(np.array([[2,3]]), np.array([[4,5]]))