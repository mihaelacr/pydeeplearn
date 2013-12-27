""" This module is manily created to test the RestrictedBoltzmannMachine
on MNIST data and see how it behaves. It is not used for classification of
handwritten digits, but rather as a way of visualizing the error of the RBM
and the weights, to see what features we have learned"""

import matplotlib.pyplot as plt
import numpy as np
# TODO: use cpikle instead of pikle
import pickle
import readmnist
import RestrictedBoltzmannMachine as RBM
import utils

from common import *

#
def visualizeWeights(weights, imgShape, tileShape):
  # return utils.tile_raster_images(weights, imgShape, tileShape, tile_spacing=(1, 1),scale_rows_to_unit_interval=False)
  return utils.tile_raster_images(weights, imgShape, tileShape, tile_spacing=(1, 1))

def main():

  # t = pickle.load( open( "weights.p", "rb" ) )

  images, labels = readmnist.read([2], dataset="training", path="MNIST")
  vectors = np.array(imagesToVectors(images))

  # Normalize the vectors to have them binary
  scaledVecs = utils.scale_to_unit_interval(vectors)

  # The number of hidden units is taken from a deep learning tutorial
  # Train the network
  # The data are the values of the images have to be normalized before being
  # presented to the network
  rbm = RBM.RBM(scaledVecs, 500, RBM.contrastiveDivergence)
  rbm.train()

  recon = rbm.reconstruct(scaledVecs[0,:])
  print recon.sum()

  print recon
  plt.imshow(vectorToImage(recon, images[0].shape), cmap=plt.cm.gray)
  plt.show()

  print rbm.weights.T
  t = visualizeWeights(rbm.weights.T, images[0].shape, (10,10))
  # TODO: add pickle behaviour to RBM
  # pickle.dump(t, open( "weights.p", "wb" ) )
  # pickle.dump(rbm.weights, open( "weights.p", "wb" ) )
  # pickle.dump(rbm.biases, open( "weights.p", "wb" ) )

  print t.sum()
  print t.shape
  plt.imshow(t, cmap=plt.cm.gray)
  plt.show()
  print "done"


if __name__ == '__main__':
  main()