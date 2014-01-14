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
import PCA

from common import *

# TODO: make these flags
NETWORK_FILE = "rbm.p"
DEEP_BELIEF_FILE = 'deepbelief_momentum.p'

parser = argparse.ArgumentParser(description='RBM for digit recognition')
parser.add_argument('--save',dest='save',action='store_true', default=False,
                    help="if true, the network is serialized and saved")
parser.add_argument('--train',dest='train',action='store_true', default=False,
                    help=("if true, the network is trained from scratch from the"
                          "traning data"))
parser.add_argument('--pca', dest='pca',action='store_true', default=False,
                    help=("if true, the code for running PCA on the data is run"))
parser.add_argument('--rbm', dest='rbm',action='store_true', default=False,
                    help=("if true, the code for traning an rbm on the data is run"))
parser.add_argument('--db', dest='deepbelief',action='store_true', default=False,
                    help=("if true, the code for traning a deepbelief net on the"
                          "data is run"))

# Get the arguments of the program
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

  trainingScaledVectors = trainVectors / 255.0
  testingScaledVectors = testingVectors / 255.0

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


def pcaOnMnist(training, dimension=700):
  res = PCA.pca(training, dimension)
  low, same = PCA.reduce(res, training)
  print "low.shape"
  print low.shape
  print same.shape

  image2DInitial = vectorToImage(training[0], (28,28))
  print same[0].shape
  image2D = vectorToImage(same[0], (28,28))

  plt.imshow(image2DInitial, cmap=plt.cm.gray)
  plt.show()
  plt.imshow(image2D, cmap=plt.cm.gray)
  plt.show()
  print "done"

def deepbeliefMain():
  # TODO: make these flags
  training = 1000
  testing = 100

  trainVectors, trainLabels =\
      readmnist.readNew(0, training, bTrain=True, path="MNIST")
  testVectors, testLabels =\
      readmnist.readNew(0, testing, bTrain=False, path="MNIST")
  print trainVectors[0].shape

  trainVectors, trainLabels = shuffle(trainVectors, trainLabels)

  trainingScaledVectors = trainVectors / 255.0
  testingScaledVectors = testVectors / 255.0

  vectorLabels = labelsToVectors(trainLabels, 10)

  if args.train:
    # net = db.DBN(3, [784, 500, 10], [Sigmoid(), Softmax()])
    # net = db.DBN(4, [784, 500, 500, 10], [Sigmoid, Sigmoid, Softmax])

    net = db.DBN(5, [784, 500, 500, 500, 10], [Sigmoid, Sigmoid, Sigmoid, Softmax])
    # TODO: think about what the network should do for 2 layers
    net.train(trainingScaledVectors, vectorLabels)
  else:
    # Take the saved network and use that for reconstructions
    f = open(DEEP_BELIEF_FILE, "rb")
    net = pickle.load(f)
    f.close()


  probs, predicted = net.classify(testingScaledVectors)
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


# think of normalizing them to 0.1 for pca as well
def pcaMain():
  training = 10000
  testing = 100

  # print args.train
  train, trainLabels =\
      readmnist.readNew(0, training, bTrain=True, path="MNIST")
  testVectors, testLabels =\
      readmnist.readNew(0, testing, bTrain=False, path="MNIST")
  print train[0].shape

  pcaOnMnist(train, dimension=100)

def main():
  if args.db + args.pca + args.rbm != 1:
    raise Exception("You decide on one main method to run")

  if args.db:
    deepbeliefMain()
  if args.pca:
    pcaMain()
  if args.rbm:
    rbmMain()


if __name__ == '__main__':
  main()
