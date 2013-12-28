""" This module is manily created to test the RestrictedBoltzmannMachine
on MNIST data and see how it behaves. It is not used for classification of
handwritten digits, but rather as a way of visualizing the error of the RBM
and the weights, to see what features we have learned"""

# TODO: use cpikle instead of pikle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle
import readmnist
import restrictedBoltzmannMachine as RBM
import utils

from common import *

NETWORK_FILE = "weights.p"

# Get the arguments of the program
parser = argparse.ArgumentParser(description='RBM for digit recognition')
parser.add_argument('--save', type=bool,
                    default=True, help="if true, the network is serialized and saved")
parser.add_argument('--train', type=bool,
                    default=True, help="if true, the network is trained from scratch from the traning data")
args = parser.parse_args()


def visualizeWeights(weights, imgShape, tileShape):
  return utils.tile_raster_images(weights, imgShape, tileShape, tile_spacing=(1, 1))

def main():
  trainImages, trainLabels = readmnist.read([2], dataset="training", path="MNIST")
  testImages, testLabels = readmnist.read([2], dataset="testing", path="MNIST")
  trainVectors = imagesToVectors(trainImages)

  # trainingScaledVectors = utils.scale_to_unit_interval(vectors)
  trainingScaledVectors = trainVectors / 256

  testingVectors = imagesToVectors(testImages)
  testingScaledVectors = testingVectors / 256

  # Train the network
  if args.train:
    # The number of hidden units is taken from a deep learning tutorial
    # The data are the values of the images have to be normalized before being
    # presented to the network
    rbm = RBM.RBM(trainingScaledVectors, 500, RBM.contrastiveDivergence)
    rbm.train()
    t = visualizeWeights(rbm.weights.T, trainImages[0].shape, (10,10))
  else:
    # Take the saved network and use that for reconstructions
    f = open(NETWORK_FILE, "rb")
    t = pickle.load(f)
    rbm = pickle.load(f)


  # Reconstruct a training image and see that it actually looks like a digit
  recon = rbm.reconstruct(testingScaledVectors[0,:])
  plt.imshow(vectorToImage(recon, trainImages[0].shape), cmap=plt.cm.gray)
  plt.show()

  # Show the weights and their form in a tile fashion
  plt.imshow(t, cmap=plt.cm.gray)
  plt.show()
  print "done"

  # TODO: add pickle behaviour to RBM
  # is this needd?
  if args.save:
    f = open("NETWORK_FILE.p", "wb")
    pickle.dump(t, f)
    pickle.dump(rbm, f)



if __name__ == '__main__':
  main()