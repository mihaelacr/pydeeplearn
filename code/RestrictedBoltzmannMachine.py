"""Implementation of restricted boltzmann machine


You need to be able to deal with different energy functions

This allows you to deal with real valued unit

do updates in parallel using multiprocessing.pool

"""



def sigmoid(x):
  return 1 / 1 + math.exp(-x);
