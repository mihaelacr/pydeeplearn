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
import restrictedBoltzmannMachine as rbm
import deepbelief as db
import utils

from common import *

NETWORK_FILE = "rbm.p"
DEEP_BELIEF_FILE = 'deepbelief_momentum.p'

# Get the arguments of the program
parser = argparse.ArgumentParser(description='RBM for digit recognition')
parser.add_argument('--save',
                    type=bool,
                    default=True,
                    help="if true, the network is serialized and saved")
parser.add_argument('--train',
                    type=bool,
                    default=True,
                    help="if true, the network is trained from scratch from the traning data")
args = parser.parse_args()


def visualizeWeights(weights, imgShape, tileShape):
  return utils.tile_raster_images(weights, imgShape,
                                  tileShape, tile_spacing=(1, 1))


def rbmMain():
  trainImages, trainLabels =\
      readmnist.readNew(0, 100, bTrain=True, path="MNIST")
  testImages, testLabels =\
      readmnist.readNew(0, 100, bTrain=False, path="MNIST")

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
    nrVisible = len(trainingScaledVectors[0])
    nrHidden = 500
    net = rbm.RBM(nrVisible, nrHidden, rbm.contrastiveDivergence)
    net.train(trainingScaledVectors)
    t = visualizeWeights(net.weights.T, trainImages[0].shape, (10,10))
  else:
    # Take the saved network and use that for reconstructions
    f = open(NETWORK_FILE, "rb")
    t = pickle.load(f)
    net = pickle.load(f)


  # Reconstruct a training image and see that it actually looks like a digit
  recon = net.reconstruct(testingScaledVectors[0,:])
  plt.imshow(vectorToImage(recon, trainImages[0,:].shape), cmap=plt.cm.gray)
  plt.show()

  # Show the weights and their form in a tile fashion
  plt.imshow(t, cmap=plt.cm.gray)
  plt.show()
  print "done"

  # TODO: add pickle behaviour to RBM
  # is this needd?
  if args.save:
    f = open(NETWORK_FILE, "wb")
    pickle.dump(t, f)
    pickle.dump(net, f)


def shuffle(data, labels):
  indexShuffle = np.random.permutation(len(data))
  shuffledData = np.array([data[i] for i in indexShuffle])
  shuffledLabels = np.array([labels[i] for i in indexShuffle])

  return shuffledData, shuffledLabels

def deepbeliefMain():
  # trainImages, trainLabels =\
  #     readmnist.read(range(10), dataset="training", path="MNIST")
  # testImages, testLabels =\
  #     readmnist.read(range(10), dataset="testing", path="MNIST")
  # TODO: make these flags
  training = 1000
  testing = 100

  trainImages, trainLabels =\
      readmnist.readNew(0, training, bTrain=True, path="MNIST")
  testImages, testLabels =\
      readmnist.readNew(0, testing, bTrain=False, path="MNIST")
  print trainImages[0].shape

  # trainVectors = imagesToVectors(trainImages)
  trainVectors, trainLabels = shuffle(trainImages, trainLabels)

  # trainingScaledVectors = utils.scale_to_unit_interval(vectors)
  trainingScaledVectors = trainVectors / 255.0

  # testingVectors = imagesToVectors(testImages)
  testingScaledVectors = testImages / 255.0

  vectorLabels = labelsToVectors(trainLabels, 10)

  if args.train:
    # net = db.DBN(3, [784, 500, 10], [Sigmoid(), Softmax()])
    net = db.DBN(4, [784, 500, 500, 10], [Tanh, Tanh, Softmax])

    # TODO: think about what the network should do for 2 layers
    net.train(trainingScaledVectors, vectorLabels)
  else:
    # Take the saved network and use that for reconstructions
    f = open(DEEP_BELIEF_FILE, "rb")
    net = pickle.load(f)


  probs, predicted = net.classify(testingScaledVectors[0: testing])
  correct = 0
  for i in xrange(testing):
    print "predicted"
    print "probs"
    print probs[i]
    print predicted[i]
    print "actual"
    actual = testLabels[i]
    print actual
    correct += (predicted[i] == actual)

  print "correct"
  print correct


  # for w in net.weights:
  #   print w

  # for b in net.biases:
  #   print b


  # t = visualizeWeights(net.weights[0].T, trainImages[0].(28, 28), (10,10))
  # plt.imshow(t, cmap=plt.cm.gray)
  # plt.show()
  # print "done"

  if args.save:
    f = open(DEEP_BELIEF_FILE, "wb")
    pickle.dump(net, f)
    f.close()


def main():
  deepbeliefMain()


if __name__ == '__main__':
  main()
