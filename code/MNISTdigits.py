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
import PCA
import svm

from sklearn import cross_validation

from common import *
from activationfunctions import *


parser = argparse.ArgumentParser(description='digit recognition')
parser.add_argument('--save',dest='save',action='store_true', default=False,
                    help="if true, the network is serialized and saved")
parser.add_argument('--train',dest='train',action='store_true', default=False,
                    help=("if true, the network is trained from scratch from the"
                          "training data"))
parser.add_argument('--sparsity', dest='sparsity',action='store_true', default=False,
                    help=("if true, the the networks are trained with sparsity constraints"))
parser.add_argument('--ann',dest='ann',action='store_true', default=False,
                    help=("if true, we train an ann not a dbn"))
parser.add_argument('--pca', dest='pca',action='store_true', default=False,
                    help=("if true, the code for running PCA on the data is run"))
parser.add_argument('--rbm', dest='rbm',action='store_true', default=False,
                    help=("if true, the code for training an rbm on the data is run"))
parser.add_argument('--rbmGauss', dest='rbmGauss',action='store_true', default=False,
                    help=("if true, the code for training an rbm on the data is run"))
parser.add_argument('--db', dest='db',action='store_true', default=False,
                    help=("if true, the code for training a deepbelief net on the"
                          "data is run"))
parser.add_argument('--dbgauss', dest='dbgauss',action='store_true', default=False,
                    help=(("if true, a dbn is trained with gaussian visible units for rbms"
                      "and relu for hidden units")))
parser.add_argument('--nesterov', dest='nesterov',action='store_true', default=False,
                    help=("if true, the deep belief net is trained using nesterov momentum"))
parser.add_argument('--rbmnesterov', dest='rbmnesterov',action='store_true', default=False,
                    help=("if true, rbms are trained using nesterov momentum"))
parser.add_argument('--rmsprop', dest='rmsprop',action='store_true', default=False,
                    help=("if true, rmsprop is used when training the deep belief net."))
parser.add_argument('--rbmrmsprop', dest='rbmrmsprop',action='store_true', default=False,
                    help=("if true, rmsprop is used when training the rbms."))
parser.add_argument('--cvgauss', dest='cvgauss',action='store_true', default=False,
                    help=("if true, performs cv on the MNIST data with gaussian units"))
parser.add_argument('--cv', dest='cv',action='store_true', default=False,
                    help=("if true, performs cv on the MNIST data"))
parser.add_argument('--display', dest='display',action='store_true', default=False,
                    help=("if true saves images of the net weights and samples from the net"))
parser.add_argument('--svm', dest='svm',action='store_true', default=False,
                    help=("if true trains an svm with the hidden activations from the net"))
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

def rbmMain(reconstructRandom=False):
  trainVectors, trainLabels =\
      readmnist.read(0, args.trainSize, digits=None, bTrain=True, path="MNIST")
  testingVectors, testLabels =\
      readmnist.read(0, args.testSize, digits=None, bTrain=False, path="MNIST")

  trainingScaledVectors = trainVectors / 255.0
  testingScaledVectors = testingVectors / 255.0

  # TODO: the reconstruction for relu still looks weird
  if args.relu:
    activationFunction = RectifiedNoisy()
    learningRate = 5e-05
    binary=False
  else:
    learningRate = 0.3
    binary=True
    activationFunction = Sigmoid()

  # Train the network
  if args.train:
    # The number of hidden units is taken from a deep learning tutorial
    # The data are the values of the images have to be normalized before being
    # presented to the network
    nrVisible = len(trainingScaledVectors[0])
    nrHidden = 500
    # use 1 dropout to test the rbm for now
    net = rbm.RBM(nrVisible, nrHidden, learningRate, 1, 1,
                  visibleActivationFunction=activationFunction,
                  hiddenActivationFunction=activationFunction,
                  rmsprop=args.rbmrmsprop, nesterov=args.rbmnesterov,
                  sparsityConstraint=args.sparsity,
                  sparsityRegularization=0.01,
                  sparsityTraget=0.01)
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

  hidden = net.hiddenRepresentation(test.reshape(1, test.shape[0]))
  plt.imshow(vectorToImage(hidden, (25,20)), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('hiddenfeatures7.png', transparent=True)


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


def makeMNISTpic():
  f = lambda x: readmnist.read(0, 1000, digits=[x], path="MNIST")[0][0].reshape((28,28))
  img = np.hstack(map(f, xrange(10)))
  print img.shape
  plt.imshow(img, cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('MNISTdigits.png', transparent=True)
  # plt.show()


def getMissclassifiedDigits():
  # First let's display the nets layer weights:
  with open(args.netFile, "rb") as f:
    dbnNet = pickle.load(f)

  testVectors, testLabels =\
      readmnist.read(0, args.testSize, digits=None, bTrain=False, path="MNIST")

  testVectors = testVectors / 255.0

  _, predictedLabels = dbnNet.classify(testVectors)

  missclassified = []
  actualLabels = []
  wrongPredictedLabels = []
  count = 0
  i = 0
  while count < 10 and i < args.testSize:
    if not predictedLabels[i] == testLabels[i]:
      missclassified += [testVectors[i].reshape((28, 28))]
      actualLabels += [testLabels[i]]
      wrongPredictedLabels += [predictedLabels[i]]
      count += 1

    i+= 1

  print "worked on " + str(i) + "examples before finding 10 wrong"

  misspreditctedimg = np.hstack(missclassified)
  plt.imshow(misspreditctedimg, cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('misspredictedMNISTdigits.png', transparent=True)

  print "predicted"
  print wrongPredictedLabels

  print "actual"
  print actualLabels

def displayWeightsAndDbSample():
  # First let's display the nets layer weights:
  with open(args.netFile, "rb") as f:
    dbnNet = pickle.load(f)

  for i in xrange(dbnNet.nrLayers - 1):
    w = dbnNet.weights[i]
    if i == 0:
      t = visualizeWeights(w.T, (28,28), (10,10))
    else:
      t = visualizeWeights(w.T, (40, 25), (10, 10))

    plt.imshow(t, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig('weightslayer' + str(i) +'.png', transparent=True)


  # then sample from the net
  samples = dbnNet.sample(10)
  reshaped = map(lambda x: x.reshape(28, 28), samples)
  reshaped = np.hstack(reshaped)

  plt.imshow(reshaped, cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('samples.png', transparent=True)

  testVectors, testLabels =\
      readmnist.read(0, args.testSize, digits=None, bTrain=False, path="MNIST")

  testVectors = testVectors / 255.0
  activationList = dbnNet.getHiddenActivations(testVectors)

  activations = activationList[-1]

  nice = []
  for activation in activationList:
    reshaped = map(lambda x: x.reshape(25, 40), activation)
    reshaped = np.hstack(reshaped)
    nice += [reshaped]

  nice = np.vstack(nice)

  plt.imshow(activations[0].reshape(25,40), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('activationsingle.png', transparent=True)


  plt.imshow(nice, cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('allactivations.png', transparent=True)


def rbmMainGauss(reconstructRandom=False):
  trainVectors, trainLabels =\
      readmnist.read(0, args.trainSize, digits=None, bTrain=True, path="MNIST")
  testVectors, testLabels =\
      readmnist.read(0, args.testSize, digits=None, bTrain=False, path="MNIST")

  trainVectors = np.array(trainVectors, dtype='float')
  trainingScaledVectors = scale(trainVectors)

  testVectors = np.array(testVectors, dtype='float')
  testingScaledVectors = scale(testVectors)

  learningRate = 0.0005
  # Train the network
  if args.train:
    # The number of hidden units is taken from a deep learning tutorial
    # The data are the values of the images have to be normalized before being
    # presented to the network
    nrVisible = len(trainingScaledVectors[0])
    nrHidden = 500
    # use 1 dropout to test the rbm for now
    net = rbm.RBM(nrVisible, nrHidden, learningRate, 1, 1,
                  visibleActivationFunction=Identity(),
                  hiddenActivationFunction=RectifiedNoisy(),
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
  plt.savefig('initial7relu.png', transparent=True)
  # plt.show()

  # Show the reconstruction
  recon = net.reconstruct(test.reshape(1, test.shape[0]))
  plt.imshow(vectorToImage(recon, (28,28)), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('reconstruct7withallrelu.png', transparent=True)

  # plt.show()

  # Show the weights and their form in a tile fashion
  # Plot the weights
  plt.imshow(t, cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('weights2srmsproprelu.png', transparent=True)
  # plt.show()

  print "done"

  if args.save:
    f = open(args.netFile, "wb")
    pickle.dump(t, f)
    pickle.dump(net, f)


def makeNicePlots():
  trainVectors, trainLabels =\
      readmnist.read(0, args.trainSize, digits=[2], bTrain=True, path="MNIST")
  testingVectors, testLabels =\
      readmnist.read(0, args.testSize, digits=[2], bTrain=False, path="MNIST")

  trainingScaledVectors = trainVectors / 255.0
  testingScaledVectors = testingVectors / 255.0

  # TODO: the reconstruction for relu still looks weird
  learningRate = 0.1
  binary = True
  activationFunction = Sigmoid()

  # Train the network
  if args.train:
    # The number of hidden units is taken from a deep learning tutorial
    # The data are the values of the images have to be normalized before being
    # presented to the network
    nrVisible = len(trainingScaledVectors[0])
    nrHidden = 500
    # use 1 dropout to test the rbm for now
    net = rbm.RBM(nrVisible, nrHidden, learningRate, 1, 1,
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

  incomingWeightVector  = net.weights.T[0]

  print " testingVectors[0]"
  print testingVectors[0]
  testVec = testingScaledVectors[0]

  # reshape this vector to be 28, 28
  reshapedWeightVector = incomingWeightVector.reshape((28, 28))
  print "reshapedWeightVector"
  print reshapedWeightVector

  reshapedTestVec = testVec.reshape((28, 28))
  print "reshapedTestVec"
  print reshapedTestVec

  overpose = reshapedWeightVector * reshapedTestVec
  print "overpose"
  print overpose

  plt.imshow(np.absolute(reshapedWeightVector), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('weightvectorreshaped.png', transparent=True)

  plt.imshow(np.absolute(reshapedTestVec), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('reshapedTestVec.png', transparent=True)

  plt.imshow(np.absolute(overpose), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('overpose.png', transparent=True)



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
  assert not args.relu, "do not run this function for rectified linear units"
  training = args.trainSize

  data, labels =\
      readmnist.read(0, training, bTrain=True, path="MNIST")

  data, labels = shuffle(data, labels)
  scaledData = data / 255.0
  vectorLabels = labelsToVectors(labels, 10)

  activationFunction = Sigmoid()


  bestFold = -1
  bestError = np.inf



  if args.relu:
    # params =[(0.01, 0.01) , (0.01, 0.05), (0.05, 0.1), (0.05, 0.05)]
    # params =[(0.0001, 0.01), (0.00001, 0.001), (0.00001, 0.0001), (0.0001, 0.1)]
    params =[(1e-05, 0.001, 0.9), (5e-06, 0.001, 0.9), (5e-05, 0.001, 0.9),
             (1e-05, 0.001, 0.95), (5e-06, 0.001, 0.95), (5e-05, 0.001, 0.95),
             (1e-05, 0.001, 0.99), (5e-06, 0.001, 0.99), (5e-05, 0.001, 0.99)]
  else:
    # params =[(0.1, 0.1) , (0.1, 0.05), (0.05, 0.1), (0.05, 0.05)]
    params =[(0.05, 0.05) , (0.05, 0.075), (0.075, 0.05), (0.075, 0.075)]
    # params =[(0.05, 0.075, 0.1), (0.05, 0.1, 0.1), (0.01, 0.05, 0.1),
    #          (0.05, 0.075, 0.01), (0.05, 0.1, 0.01), (0.01, 0.05, 0.01),
    #          (0.05, 0.075, 0.001), (0.05, 0.1, 0.001), (0.01, 0.05, 0.001)]

  nrFolds = len(params)
  kf = cross_validation.KFold(n=training, n_folds=nrFolds)

  i = 0
  for training, testing in kf:
    # Train the net
    # Try 1200, 1200, 1200

    trainData = scaledData[training]
    trainLabels = vectorLabels[training]

    # net = db.DBN(5, [784, 1000, 1000, 1000, 10],
    net = db.DBN(5, [784, 500, 500, 2000, 10],
                  binary=1-args.relu,
                  unsupervisedLearningRate=params[i][0],
                  supervisedLearningRate=params[i][1],
                  momentumMax=0.95,
                  nesterovMomentum=args.nesterov,
                  rbmNesterovMomentum=args.rbmnesterov,
                  activationFunction=activationFunction,
                  rbmActivationFunctionVisible=activationFunction,
                  rbmActivationFunctionHidden=activationFunction,
                  rmsprop=args.rmsprop,
                  visibleDropout=0.8,
                  hiddenDropout=0.5,
                  weightDecayL1=0,
                  weightDecayL2=0,
                  rbmHiddenDropout=1.0,
                  rbmVisibleDropout=1.0,
                  miniBatchSize=args.miniBatchSize,
                  preTrainEpochs=args.preTrainEpochs,
                  sparsityTragetRbm=0.01,
                  sparsityConstraintRbm=False,
                  sparsityRegularizationRbm=None)

    net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
              validation=args.validation)

    proabilities, predicted = net.classify(scaledData[testing])

    testLabels = labels[testing]
    # Test it with the testing data and measure the missclassification error
    error = getClassificationError(predicted, testLabels)

    print "error for " + str(params[i])
    print error

    if error < bestError:
      bestError = error
      bestFold = i
    i += 1

  print "best fold was " + str(bestFold)
  print "bestParameter " + str(params[bestFold])
  print "bestError" + str(bestError)



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

  if args.save:
    f = open(args.netFile, "wb")
    pickle.dump(net, f)
    f.close()

def svmMNIST():
  with open(args.netFile, "rb") as f:
    dbnNet = pickle.load(f)

  trainVectors, trainLabels =\
      readmnist.read(0, args.trainSize, bTrain=True, path="MNIST")

  testVectors, testLabels =\
      readmnist.read(0, args.testSize, digits=None, bTrain=False, path="MNIST")

  trainingScaledVectors = trainVectors / 255.0
  testingScaledVectors = testVectors / 255.0

  svm.SVMCV(dbnNet, trainingScaledVectors, trainLabels,
            testingScaledVectors, testLabels)

# NOT for relu: use GaussianMNIST for that
def deepbeliefMNIST():

  assert not args.relu, "do not run this method for rectified linear units"

  training = args.trainSize
  testing = args.testSize

  trainVectors, trainLabels =\
      readmnist.read(0, training, bTrain=True, path="MNIST")
  testVectors, testLabels =\
      readmnist.read(0, testing, bTrain=False, path="MNIST")
  print trainVectors[0].shape

  trainVectors, trainLabels = shuffle(trainVectors, trainLabels)

  activationFunction = Sigmoid()

  # TODO: do not divide for RELU?
  trainingScaledVectors = trainVectors / 255.0
  testingScaledVectors = testVectors / 255.0

  vectorLabels = labelsToVectors(trainLabels, 10)

  unsupervisedLearningRate = 0.01
  supervisedLearningRate = 0.05
  momentumMax = 0.95

  if args.train:
    net = db.DBN(5, [784, 1000, 1000, 1000, 10],
                 binary=False,
                 unsupervisedLearningRate=unsupervisedLearningRate,
                 supervisedLearningRate=supervisedLearningRate,
                 momentumMax=momentumMax,
                 activationFunction=activationFunction,
                 rbmActivationFunctionVisible=activationFunction,
                 rbmActivationFunctionHidden=activationFunction,
                 nesterovMomentum=args.nesterov,
                 rbmNesterovMomentum=args.rbmnesterov,
                 rmsprop=args.rmsprop,
                 hiddenDropout=0.5,
                 visibleDropout=0.8,
                 rbmHiddenDropout=1.0,
                 rbmVisibleDropout=1.0,
                 weightDecayL1=0,
                 weightDecayL2=0,
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

def deepbeliefMNISTGaussian():
  training = args.trainSize
  testing = args.testSize

  trainVectors, trainLabels =\
      readmnist.read(0, training, bTrain=True, path="MNIST")
  testVectors, testLabels =\
      readmnist.read(0, testing, bTrain=False, path="MNIST")
  print trainVectors[0].shape

  trainVectors, trainLabels = shuffle(trainVectors, trainLabels)

  trainVectors = np.array(trainVectors, dtype='float')
  trainingScaledVectors = scale(trainVectors)

  testVectors = np.array(testVectors, dtype='float')
  testingScaledVectors = scale(testVectors)

  vectorLabels = labelsToVectors(trainLabels, 10)

  unsupervisedLearningRate = 0.005
  supervisedLearningRate = 0.005
  momentumMax = 0.95
  sparsityTragetRbm = 0.01
  sparsityConstraintRbm = False
  sparsityRegularizationRbm = 0.05

  if args.train:
    net = db.DBN(5, [784, 1200, 1200, 1200, 10],
                 binary=False,
                 unsupervisedLearningRate=unsupervisedLearningRate,
                 supervisedLearningRate=supervisedLearningRate,
                 momentumMax=momentumMax,
                 activationFunction=Rectified(),
                 rbmActivationFunctionVisible=Identity(),
                 rbmActivationFunctionHidden=RectifiedNoisy(),
                 nesterovMomentum=args.nesterov,
                 rbmNesterovMomentum=args.rbmnesterov,
                 rmsprop=args.rmsprop,
                 hiddenDropout=0.5,
                 visibleDropout=0.8,
                 rbmHiddenDropout=1.0,
                 rbmVisibleDropout=1.0,
                 weightDecayL1=0,
                 weightDecayL2=0,
                 sparsityTragetRbm=sparsityTragetRbm,
                 sparsityConstraintRbm=sparsityConstraintRbm,
                 sparsityRegularizationRbm=sparsityRegularizationRbm,
                 preTrainEpochs=args.preTrainEpochs)

    net.train(trainingScaledVectors, vectorLabels,
              maxEpochs=args.maxEpochs, validation=args.validation)
  else:
    # Take the saved network and use that for reconstructions
    f = open(args.netFile, "rb")
    net = pickle.load(f)
    f.close()


  probs, predicted = net.classify(testingScaledVectors)
  print type(predicted)
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

  if args.save:
    f = open(args.netFile, "wb")
    pickle.dump(net, f)
    f.close()

def cvMNISTGaussian():
  training = args.trainSize

  trainVectors, trainLabels =\
      readmnist.read(0, training, bTrain=True, path="MNIST")

  trainVectors, trainLabels = shuffle(trainVectors, trainLabels)

  trainVectors = np.array(trainVectors, dtype='float')

  # Ensure the data has zero mean and 1 variance
  trainingScaledVectors = scale(trainVectors)
  vectorLabels = labelsToVectors(trainLabels, 10)

  bestFold = -1
  bestError = np.inf

  params = [(5e-03, 1e-02),  (1e-02, 5e-02), (5e-03, 5e-02), (1e-02, 5e-03), (5e-03, 5e-03), (1e-02, 1e-02) ]

  correctness = []

  nrFolds = len(params)

  kf = cross_validation.KFold(n=training, n_folds=nrFolds)

  i = 0
  for train, test in kf:
    # Train the net
    # Try 1200, 1200, 1200
    net = db.DBN(5, [784, 1000, 1000, 1000, 10],
                  binary=False,
                  unsupervisedLearningRate=params[i][0],
                  supervisedLearningRate=params[i][1],
                  momentumMax=0.95,
                  nesterovMomentum=args.nesterov,
                  rbmNesterovMomentum=args.rbmnesterov,
                  activationFunction=Rectified(),
                  rbmActivationFunctionVisible=Identity(),
                  rbmActivationFunctionHidden=RectifiedNoisy(),
                  rmsprop=args.rmsprop,
                  visibleDropout=0.8,
                  hiddenDropout=0.5,
                  weightDecayL1=0,
                  weightDecayL2=0,
                  rbmHiddenDropout=1.0,
                  rbmVisibleDropout=1.0,
                  miniBatchSize=args.miniBatchSize,
                  preTrainEpochs=args.preTrainEpochs,
                  sparsityConstraintRbm=False,
                  sparsityTragetRbm=0.01,
                  sparsityRegularizationRbm=None)

    net.train(trainingScaledVectors[train], vectorLabels[train],
              maxEpochs=args.maxEpochs,
              validation=args.validation)

    proabilities, predicted = net.classify(trainingScaledVectors[test])
    # Test it with the testing data and measure the missclassification error
    error = getClassificationError(predicted, trainLabels[test])

    print "error for " + str(params[i])
    print error

    correct = 1.0 - error

    if error < bestError:
      bestError = error
      bestFold = i

    i += 1

    correctness += [correct]

  print "best fold was " + str(bestFold)
  print "bestParameter " + str(params[bestFold])
  print "bestError " + str(bestError)

  for i in xrange(len(params)):
    print "parameter tuple " + str(params[i]) + " achieved correctness of " + str(correctness[i])



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

def main():
  import random
  print "FIXING RANDOMNESS"
  random.seed(6)
  np.random.seed(6)
  if args.db + args.pca + args.rbm + args.cv +\
      args.ann + args.cvgauss + args.rbmGauss + args.dbgauss + args.display + args.svm != 1:
    raise Exception("You have to decide on one main method to run")

  # makeNicePlots()
  # makeMNISTpic()
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
  if args.cvgauss:
    cvMNISTGaussian()
  if args.rbmGauss:
    rbmMainGauss()
  if args.dbgauss:
    deepbeliefMNISTGaussian()
  if args.display:
    displayWeightsAndDbSample()
  if args.svm:
    svmMNIST()


if __name__ == '__main__':
  main()
