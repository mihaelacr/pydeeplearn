""" This module is manily created to test the deep belief and
rbm implementations on MNIST"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import readmnist
import restrictedBoltzmannMachine as rbm
import deepbelief as db
import ann
import utils
import PCA

from sklearn import lda
from sklearn.decomposition import PCA

import DimensionalityReduction

from common import *

parser = argparse.ArgumentParser(description='digit recognition')
parser.add_argument('--save',dest='save',action='store_true', default=False,
                    help="if true, the network is serialized and saved")
parser.add_argument('--train',dest='train',action='store_true', default=False,
                    help=("if true, the network is trained from scratch from the"
                          "traning data"))
parser.add_argument('--ann',dest='ann',action='store_true', default=False,
                    help=("if true, we train an ann not a dbn"))
parser.add_argument('--pca', dest='pca',action='store_true', default=False,
                    help=("if true, the code for running PCA on the data is run"))
parser.add_argument('--rbm', dest='rbm',action='store_true', default=False,
                    help=("if true, the code for traning an rbm on the data is run"))
parser.add_argument('--db', dest='db',action='store_true', default=False,
                    help=("if true, the code for traning a deepbelief net on the"
                          "data is run"))
parser.add_argument('--lda', dest='lda',action='store_true', default=False,
                    help=("if true, runs LDA main"))
parser.add_argument('--pcadbn', dest='pcadbn',action='store_true', default=False,
                    help=("if true, runs dbn with pca performed first main"))
parser.add_argument('--nesterov', dest='nesterov',action='store_true', default=False,
                    help=("if true, the deep belief net is trained using nesterov momentum"))
parser.add_argument('--rbmnesterov', dest='rbmnesterov',action='store_true', default=False,
                    help=("if true, rbms are trained using nesterov momentum"))
parser.add_argument('--rmsprop', dest='rmsprop',action='store_true', default=False,
                    help=("if true, rmsprop is used when training the deep belief net."))
parser.add_argument('--rbmrmsprop', dest='rbmrmsprop',action='store_true', default=False,
                    help=("if true, rmsprop is used when training the rbms."))
parser.add_argument('--cv', dest='cv',action='store_true', default=False,
                    help=("if true, performs cv on the MNIST data"))
parser.add_argument('--relu', dest='relu',action='store_true', default=False,
                    help=("if true, trains the RBM or DBN with a rectified linear unit"))
parser.add_argument('--trainSize', type=int, default=10000,
                    help='the number of tranining cases to be considered')
parser.add_argument('--testSize', type=int, default=1000,
                    help='the number of testing cases to be considered')
parser.add_argument('--preTrainEpochs', type=int, default=1,
                    help='the number of pretraining epochs')
parser.add_argument('--maxEpochs', type=int, default=100,
                    help='the maximum number of supervised epochs')
parser.add_argument('--miniBatchSize', type=int, default=10,
                    help='the number of training points in a mini batch')
parser.add_argument('netFile', help="file where the serialized network should be saved")
parser.add_argument('--validation',dest='validation',action='store_true', default=False,
                    help="if true, the network is trained using a validation set")


# DEBUG mode?
parser.add_argument('--debug', dest='debug',action='store_true', default=False,
                    help=("if true, the deep belief net is ran in DEBUG mode"))

# Get the arguments of the program
args = parser.parse_args()

# Set the debug mode in the deep belief net
db.DEBUG = args.debug

BINARY = {T.nnet.sigmoid : True}.get(False)

def rbmMain(reconstructRandom=False):
  trainVectors, trainLabels =\
      readmnist.read(0, args.trainSize, digits=[2], bTrain=True, path="MNIST")
  testingVectors, testLabels =\
      readmnist.read(0, args.testSize, digits=None, bTrain=False, path="MNIST")

  trainingScaledVectors = trainVectors / 255.0
  testingScaledVectors = testingVectors / 255.0


  if args.relu:
    activationFunction = relu
  else:
    activationFunction = T.nnet.sigmoid


  # Train the network
  if args.train:
    # The number of hidden units is taken from a deep learning tutorial
    # The data are the values of the images have to be normalized before being
    # presented to the network
    nrVisible = len(trainingScaledVectors[0])
    nrHidden = 500
    # use 1 dropout to test the rbm for now
    net = rbm.RBM(nrVisible, nrHidden, 0.01, 1, 1,
                  BINARY[activationFunction],
                  visibleActivationFunction=activationFunction,
                  hiddenActivationFunction=activationFunction,
                  rmsprop=args.rbmrmsprop, nesterov=args.rbmnesterov)
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
  plt.axis('off')
  plt.savefig('initial7.png', transparent=True)
  # plt.show()

  # Show the reconstruction
  recon = net.reconstruct(test.reshape(1, test.shape[0]))
  plt.imshow(vectorToImage(recon, (28,28)), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('reconstruct7withall.png', transparent=True)

  # plt.show()

  # Show the weights and their form in a tile fashion
  # Plot the weights
  plt.imshow(t, cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('weights2srmsprop.png', transparent=True)
  # plt.show()

  print "done"

  if args.save:
    f = open(args.netFile, "wb")
    pickle.dump(t, f)
    pickle.dump(net, f)


def pcaSklearn(training, dimension=700):
  pca = PCA(n_components=dimension)
  pca.fit(training)
  low = pca.transform(training)
  same = pca.inverse_transform(low)

  print "low[0].shape"
  print low[0].shape

  image2DInitial = vectorToImage(training[0], (28,28))
  print same[0].shape
  image2D = vectorToImage(same[0], (28,28))

  image2DLow = vectorToImage(low[0], (20,20))
  plt.imshow(image2DLow, cmap=plt.cm.gray)
  plt.show()


  plt.imshow(image2DInitial, cmap=plt.cm.gray)
  plt.show()
  plt.imshow(image2D, cmap=plt.cm.gray)
  plt.show()
  print "done"
  return low

def pcaOnMnist(training, dimension=700):
  mean, principalComponents = PCA.pca(training, dimension)
  low, same = PCA.reduce(principalComponents, training, mean, noSame=False)

  print "low[0].shape"
  print low[0].shape

  image2DInitial = vectorToImage(training[0], (28,28))
  print same[0].shape
  image2D = vectorToImage(same[0], (28,28))

  image2DLow = vectorToImage(low[0], (20,20))
  plt.imshow(image2DLow, cmap=plt.cm.gray)
  plt.show()


  plt.imshow(image2DInitial, cmap=plt.cm.gray)
  plt.show()
  plt.imshow(image2D, cmap=plt.cm.gray)
  plt.show()
  print "done"
  return low


def cvMNIST():
  training = args.trainSize
  testing = args.testSize

  trainVectors, trainLabels =\
      readmnist.read(0, training, bTrain=True, path="MNIST")
  testVectors, testLabels =\
      readmnist.read(0, testing, bTrain=False, path="MNIST")
  print trainVectors[0].shape

  trainVectors, trainLabels = shuffle(trainVectors, trainLabels)

  if args.relu:
    activationFunction = relu
  else:
    activationFunction = T.nnet.sigmoid


  trainingScaledVectors = trainVectors / 255.0
  testingScaledVectors = testVectors / 255.0

  vectorLabels = labelsToVectors(trainLabels, 10)

  permutation = np.random.permutation(range(training))
  bestFold = -1
  bestError = np.inf

  if args.relu:
    # params =[(0.01, 0.01) , (0.01, 0.05), (0.05, 0.1), (0.05, 0.05)]
    # params =[(0.0001, 0.01), (0.00001, 0.001), (0.00001, 0.0001), (0.0001, 0.1)]
    params =[(1e-05, 0.001), (5e-06, 0.001), (5e-05, 0.001)]
  else:
    params =[(0.1, 0.1) , (0.1, 0.05), (0.05, 0.1), (0.05, 0.05)]

  nrFolds = len(params)
  foldSize = training / nrFolds

  for i in xrange(nrFolds):
    # Train the net
    # Try 1200, 1200, 1200
    net = db.DBN(5, [784, 1000, 1000, 1000, 10],
                  BINARY[activationFunction],
                  unsupervisedLearningRate=params[i][0],
                  supervisedLearningRate=params[i][1],
                  nesterovMomentum=args.nesterov,
                  rbmNesterovMomentum=args.rbmnesterov,
                  activationFunction=activationFunction,
                  rmsprop=args.rmsprop,
                  hiddenDropout=0.5,
                  rbmHiddenDropout=0.5,
                  visibleDropout=0.8,
                  rbmVisibleDropout=0.9,
                  miniBatchSize=args.miniBatchSize,
                  preTrainEpochs=args.preTrainEpochs)
    foldIndices = permutation[i * foldSize : (i + 1) * foldSize - 1]
    net.train(trainingScaledVectors[foldIndices], vectorLabels[foldIndices],
              maxEpochs=args.maxEpochs,
              validation=args.validation)

    proabilities, predicted = net.classify(testingScaledVectors)
    # Test it with the testing data and measure the missclassification error
    error = getClassificationError(predicted, testLabels)

    print "error for " + str(params[i])
    print error

    if error < bestError:
      bestError = error
      bestFold = i

  print "best fold was " + str(bestFold)
  print "bestParameter " + str(params[bestFold])
  print "bestError" + str(bestError)


def getClassificationError(predicted, actual):
  return 1.0 - (predicted == actual).sum() * 1.0 / len(actual)

def annMNIST():
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
    # Try 1200, 1200, 1200
    # [784, 500, 500, 2000, 10
    net = ann.ANN(5, [784, 1000, 1000, 1000, 10],
                 supervisedLearningRate=0.001,
                 nesterovMomentum=args.nesterov,
                 rmsprop=args.rmsprop,
                 hiddenDropout=0.5,
                 visibleDropout=0.8,
                 miniBatchSize=args.miniBatchSize,
                 normConstraint=15)
    net.train(trainingScaledVectors, vectorLabels,
              maxEpochs=args.maxEpochs, validation=args.validation)
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


def deepbeliefMNIST():
  training = args.trainSize
  testing = args.testSize

  trainVectors, trainLabels =\
      readmnist.read(0, training, bTrain=True, path="MNIST")
  testVectors, testLabels =\
      readmnist.read(0, testing, bTrain=False, path="MNIST")
  print trainVectors[0].shape

  trainVectors, trainLabels = shuffle(trainVectors, trainLabels)

  if args.relu:
    activationFunction = makeNoisyRelu()
  else:
    activationFunction = T.nnet.sigmoid

  # TODO: do not divide for RELU?
  trainingScaledVectors = trainVectors / 255.0
  testingScaledVectors = testVectors / 255.0

  vectorLabels = labelsToVectors(trainLabels, 10)

  if args.relu:

    unsupervisedLearningRate = 5e-06
    supervisedLearningRate = 0.001
  else:
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.05


  if args.train:
    # Try 1200, 1200, 1200
    # [784, 500, 500, 2000, 10
    net = db.DBN(5, [784, 1000, 1000, 1000, 10],
                 BINARY[activationFunction],
                 unsupervisedLearningRate=unsupervisedLearningRate,
                 supervisedLearningRate=supervisedLearningRate,
                 activationFunction=activationFunction,
                 nesterovMomentum=args.nesterov,
                 rbmNesterovMomentum=args.rbmnesterov,
                 rmsprop=args.rmsprop,
                 hiddenDropout=0.5,
                 rbmHiddenDropout=0.5,
                 visibleDropout=0.8,
                 rbmVisibleDropout=0.9,
                 preTrainEpochs=args.preTrainEpochs)
    net.train(trainingScaledVectors, vectorLabels,
              maxEpochs=args.maxEpochs, validation=args.validation)
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

  pcaSklearn(train, dimension=400)

def pcadbn(dimension=69):
  training = args.trainSize
  testing = args.testSize

  trainVectors, trainLabels =\
      readmnist.read(0, training, bTrain=True, path="MNIST")
  testVectors, testLabels =\
      readmnist.read(0, testing, bTrain=False, path="MNIST")

  trainVectors, trainLabels = shuffle(trainVectors, trainLabels)
  # Should you rescale here as well? Probably not
  trainingScaledVectors = trainVectors
  testingScaledVectors = testVectors

  pca = PCA(n_components=dimension)
  pca.fit(trainingScaledVectors)
  reducedTrain = pca.transform(trainingScaledVectors)
  reducedTest = pca.transform(testingScaledVectors)

  # mean, principalComponents = PCA.pca(trainingScaledVectors, dimension)
  # # If we do not have enough data we have to reduce the dimension
  # # to how many non zero eigen vector we have
  # dimension = len(principalComponents)
  # reducedTrain, _ = PCA.reduce(principalComponents, trainingScaledVectors, mean)

  # reducedTest, _ = PCA.reduce(principalComponents, testingScaledVectors, mean)
  # scaledTestPCA = []

  scaled  = []
  for x in reducedTest:
    # Scale the results so that they are in between 0 and 1
    scaled += [utils.scale_to_unit_interval(x)]

  reducedTest = np.array(scaled)

  scaled  = []
  for x in reducedTrain:
    # Scale the results so that they are in between 0 and 1
    scaled += [utils.scale_to_unit_interval(x)]

  reducedTrain = np.array(scaled)


  vectorLabels = labelsToVectors(trainLabels, 10)

  if args.train:
    # Try 1200, 1200, 1200
    # [784, 500, 500, 2000, 10
    net = db.DBN(5, [dimension, 200, 200, 200, 10],
                 BINARY[activationFunction],
                 unsupervisedLearningRate=0.05,
                 supervisedLearningRate=0.05,
                 nesterovMomentum=args.nesterov,
                 rbmNesterovMomentum=args.rbmnesterov,
                 rmsprop=args.rmsprop,
                 hiddenDropout=0.5,
                 rbmHiddenDropout=0.5,
                 visibleDropout=0.8,
                 rbmVisibleDropout=0.9,
                 preTrainEpochs=args.preTrainEpochs)
    net.train(reducedTrain, vectorLabels,
              maxEpochs=args.maxEpochs, validation=args.validation)
  else:
    # Take the saved network and use that for reconstructions
    f = open(args.netFile, "rb")
    net = pickle.load(f)
    f.close()


  probs, predicted = net.classify(reducedTest)
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


# TODO: fix? or check
def ldaMain():
  training = args.trainSize
  testing = args.testSize

  train, trainLabels =\
      readmnist.read(0, training, bTrain=True, path="MNIST")
  testVectors, testLabels =\
      readmnist.read(0, testing, bTrain=False, path="MNIST")
  print train[0].shape

  train, trainLabels = shuffle(train, trainLabels)

  # You need to transpose the data, here we have the data on rows
  transposedTrain = train.T
  principalVectors = DimensionalityReduction.LDA(transposedTrain, trainLabels, dimension=700)
  meanData = transposedTrain.mean(axis=1)

  # Project
  print principalVectors.shape
  print "principalVectors.shape"
  print "testVectors.T - meanData[:, np.newaxis])"
  print (testVectors.T - meanData[:, np.newaxis]).shape
  projections = np.dot(principalVectors.T, testVectors.T - meanData[:, np.newaxis])

  # Reconstruct
  reconstructions = np.dot(principalVectors, projections) + meanData[:, np.newaxis]

  projections = projections.T

  reconstructions = reconstructions.T

  image2DInitial = vectorToImage(testVectors[0], (28,28))
  image2Dreconstruction = vectorToImage(reconstructions[0], (28,28))

  plt.imshow(image2DInitial, cmap=plt.cm.gray)
  plt.show()
  plt.imshow(image2Dreconstruction, cmap=plt.cm.gray)
  plt.show()
  print "done"

def main():
  import random
  print "FIXING RANDOMNESS"
  random.seed(6)
  np.random.seed(6)
  if args.db + args.pca + args.rbm + args.cv + args.ann + args.lda + args.pcadbn!= 1:
    raise Exception("You decide on one main method to run")

  if args.db:
    deepbeliefMNIST()
  if args.pca:
    pcaMain()
  if args.rbm:
    rbmMain()
  if args.cv:
    cvMNIST()
  if args.ann:
    annMNIST()
  if args.lda:
    ldaMain()
  if args.pcadbn:
    pcadbn()

if __name__ == '__main__':
  main()
