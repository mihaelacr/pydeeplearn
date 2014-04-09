""" This module is manily created to test the deep belief and
rbm implementations on MNIST"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import readmnist
import restrictedBoltzmannMachine as rbm
import deepbelief as db
import utils
import PCA
from sklearn import cross_validation

from common import *

parser = argparse.ArgumentParser(description='digit recognition')
parser.add_argument('--save',dest='save',action='store_true', default=False,
                    help="if true, the network is serialized and saved")
parser.add_argument('--train',dest='train',action='store_true', default=False,
                    help=("if true, the network is trained from scratch from the"
                          "traning data"))
parser.add_argument('--pca', dest='pca',action='store_true', default=False,
                    help=("if true, the code for running PCA on the data is run"))
parser.add_argument('--rbm', dest='rbm',action='store_true', default=False,
                    help=("if true, the code for traning an rbm on the data is run"))
parser.add_argument('--db', dest='db',action='store_true', default=False,
                    help=("if true, the code for traning a deepbelief net on the"
                          "data is run"))
parser.add_argument('--cv', dest='cv',action='store_true', default=False,
                    help=("if true, performs cv on the MNIST data"))
parser.add_argument('--trainSize', type=int, default=10000,
                    help='the number of tranining cases to be considered')
parser.add_argument('--testSize', type=int, default=1000,
                    help='the number of testing cases to be considered')
parser.add_argument('netFile', help="file where the serialized network should be saved")

# DEBUG mode?
parser.add_argument('--debug', dest='debug',action='store_true', default=False,
                    help=("if true, the deep belief net is ran in DEBUG mode"))

# Get the arguments of the program
args = parser.parse_args()

# Set the debug mode in the deep belief net
db.DEBUG = args.debug

def visualizeWeights(weights, imgShape, tileShape):
  return utils.tile_raster_images(weights, imgShape,
                                  tileShape, tile_spacing=(1, 1))

def rbmMain(reconstructRandom=True):
  trainVectors, trainLabels =\
      readmnist.read(0, args.trainSize, digits=None, bTrain=True, path="MNIST")
  testingVectors, testLabels =\
      readmnist.read(0, args.testSize, digits=None, bTrain=False, path="MNIST")

  trainingScaledVectors = trainVectors / 255.0
  testingScaledVectors = testingVectors / 255.0

  # Train the network
  if args.train:
    # The number of hidden units is taken from a deep learning tutorial
    # The data are the values of the images have to be normalized before being
    # presented to the network
    nrVisible = len(trainingScaledVectors[0])
    nrHidden = 500
    # use 1 dropout to test the rbm for now
    net = rbm.RBM(nrVisible, nrHidden, rbm.contrastiveDivergence, 1, 1)
    net.train(trainingScaledVectors)
    t = visualizeWeights(net.weights.T, (28,28), (10,10))
  else:
    # Take the saved network and use that for reconstructions
    f = open(args.netFile, "rb")
    t = pickle.load(f)
    net = pickle.load(f)
    f.close()

  # Reconstruct an image and see that it actually looks like a digit
  test = testingScaledVectors[0,:]

  # get a random image and see it looks like
  if reconstructRandom:
    test = np.random.random_sample(test.shape)


  # Show the initial image first
  recon = net.reconstruct(test.reshape(1, test.shape[0]))
  plt.imshow(vectorToImage(test, (28,28)), cmap=plt.cm.gray)
  plt.show()

  # Show the reconstruction
  recon = net.reconstruct(test.reshape(1, test.shape[0]))
  plt.imshow(vectorToImage(recon, (28,28)), cmap=plt.cm.gray)
  plt.show()

  # Show the weights and their form in a tile fashion
  # Plot the weights
  plt.imshow(t, cmap=plt.cm.gray)
  plt.show()
  print "done"

  if args.save:
    f = open(args.netFile, "wb")
    pickle.dump(t, f)
    pickle.dump(net, f)


def shuffle(data, labels):
  indexShuffle = np.random.permutation(len(data))
  shuffledData = np.array([data[i] for i in indexShuffle])
  shuffledLabels = np.array([labels[i] for i in indexShuffle])

  return shuffledData, shuffledLabels


def pcaOnMnist(training, dimension=700):
  principalComponents = PCA.pca(training, dimension)
  low, same = PCA.reduce(principalComponents, training)

  image2DInitial = vectorToImage(training[0], (28,28))
  print same[0].shape
  image2D = vectorToImage(same[0], (28,28))

  plt.imshow(image2DInitial, cmap=plt.cm.gray)
  plt.show()
  plt.imshow(image2D, cmap=plt.cm.gray)
  plt.show()
  print "done"


def cvMNIST():
  training = args.trainSize
  testing = args.testSize

  trainVectors, trainLabels =\
      readmnist.read(0, training, bTrain=True, path="MNIST")
  testVectors, testLabels =\
      readmnist.read(0, testing, bTrain=False, path="MNIST")
  print trainVectors[0].shape

  trainVectors, trainLabels = shuffle(trainVectors, trainLabels)

  trainingScaledVectors = trainVectors / 255.0
  testingScaledVectors = testVectors / 255.0

  vectorLabels = labelsToVectors(trainLabels, 10)

  nrFolds = 3

  permutation = np.random.permutation(range(training))
  foldSize = training / nrFolds
  bestFold = -1
  bestError = np.inf
  params = [0.01, 0.001, 0.0001]
  for i in xrange(nrFolds):
    # Train the net
    net = db.DBN(5, [784, 1000, 1000, 1000, 10],
                 [Sigmoid, Sigmoid, Sigmoid, Softmax],
                  supervisedLearningRate=params[i],
                  dropout=0.5, rbmDropout=0.5, visibleDropout=0.8,
                  rbmVisibleDropout=1)
    foldIndices = permutation[i * foldSize : (i + 1) * foldSize - 1]
    net.train(trainingScaledVectors[foldIndices], vectorLabels[foldIndices])

    proabilities, predicted = net.classify(testingScaledVectors)
    # Test it with the testing data and measure the missclassification error
    error = getClassificationError(labelsToVectors(testLabels, 10), proabilities)

    if error < bestError:
      bestError = error
      bestFold = i

  print "best fold was" + str(bestFold)
  print "bestParameter" + str(params[bestFold])


def getClassificationError(actual, probs):
  return 1.0 - (actual == probs).sum() * 1.0 / len(actual)

def deepbeliefMNIST():
  import random
  print "FIXING RANDOMNESS"
  random.seed(6)
  np.random.seed(6)

  training = args.trainSize
  testing = args.testSize

  trainVectors, trainLabels =\
      readmnist.read(0, training, bTrain=True, path="MNIST")
  testVectors, testLabels =\
      readmnist.read(0, testing, bTrain=False, path="MNIST")
  print trainVectors[0].shape

  trainVectors, trainLabels = shuffle(trainVectors, trainLabels)

  trainingScaledVectors = trainVectors / 255.0
  testingScaledVectors = testVectors / 255.0

  vectorLabels = labelsToVectors(trainLabels, 10)

  if args.train:
    # net = db.DBN(3, [784, 500, 10], [Sigmoid(), Softmax()])
    # net = db.DBN(4, [784, 500, 500, 10], [Sigmoid, Sigmoid, Softmax])

    net = db.DBN(5, [784, 1000, 1000, 1000, 10],
                 [Sigmoid, Sigmoid, Sigmoid, Softmax],
                 supervisedLearningRate=0.01,
                 dropout=0.5, rbmDropout=0.5, visibleDropout=0.8,
                 rbmVisibleDropout=1)
    # TODO: think about what the network should do for 2 layers
    net.train(trainingScaledVectors, vectorLabels)
  else:
    # Take the saved network and use that for reconstructions
    f = open(args.netFile, "rb")
    net = pickle.load(f)
    f.close()


  probs, predicted = net.classify(testingScaledVectors)
  correct = 0
  errorCases = []
  for i in xrange(testing):
    print "predicted"
    print "probs"
    print probs[i]
    print predicted[i]
    print "actual"
    actual = testLabels[i]
    print actual
    if predicted[i] == actual:
      correct += 1
    else:
      errorCases.append(i)

  # Mistakes for digits
  # You just need to display some for the report
  # trueDigits = testLabels[errorCases]
  # predictedDigits = predicted[errorCases]



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
    f = open(args.netFile, "wb")
    pickle.dump(net, f)
    f.close()

# TODO: fix this (look at the ML coursework for it)
# Even better, use LDA
# think of normalizing them to 0.1 for pca as well
def pcaMain():
  training = args.trainSize
  testing = args.testSize

  train, trainLabels =\
      readmnist.read(0, training, bTrain=True, path="MNIST")
  testVectors, testLabels =\
      readmnist.read(0, testing, bTrain=False, path="MNIST")
  print train[0].shape

  pcaOnMnist(train, dimension=100)

def main():
  if args.db + args.pca + args.rbm + args.cv!= 1:
    raise Exception("You decide on one main method to run")

  if args.db:
    deepbeliefMNIST()
  if args.pca:
    pcaMain()
  if args.rbm:
    rbmMain()
  if args.cv:
    cvMNIST()


if __name__ == '__main__':
  main()
