""" This module is manily created to test the RestrictedBoltzmannMachine
on MNIST data and see how it behaves. It is not used for classification of
handwritten digits, but rather as a way of visualizing the error of the RBM
and the weights, to see what features we have learned"""

import readmnist
import matplotlib.pyplot as plt
import numpy as np

import RestrictedBoltzmannMachine as RBM

from common import *

def main():
  images, labels = readmnist.read([2], dataset="training", path="MNIST")
  # plt.imshow(images[0], cmap=plt.cm.gray)
  # plt.show()
  vectors = np.array(imagesToVectors(images))

  rbm = RBM.RBM(vectors, 3, RBM.contrastiveDivergence)
  rbm.train()


  print "done"

if __name__ == '__main__':
  main()