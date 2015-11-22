""" The aim of this file is to contain all the functions
and the main which have to do with emotion recognition, especially
with the Kanade and Multi PiE databases. Note that the versions of both databases I used
the faces were already aligned. Those databases are not publicly available. """

__author__ = "Mihaela Rosca"
__contact__ = "mihaela.c.rosca@gmail.com"

import argparse
import cPickle as pickle
import warnings
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib
import warnings
import numpy as np

from lib import deepbelief as db
from lib import restrictedBoltzmannMachine as rbm

from lib.activationfunctions import *
from lib.common import *
from read.readfacedatabases import *

import matplotlib
import os
havedisplay = "DISPLAY" in os.environ
if not havedisplay:
  exitval = os.system('python -c "import matplotlib.pyplot as plt; plt.figure()"')
  havedisplay = (exitval == 0)
if havedisplay:
  import matplotlib.pyplot as plt
else:
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

ImportMatplotlibPlot()

parser = argparse.ArgumentParser(description='emotion recongnition')
parser.add_argument('--rbmnesterov', dest='rbmnesterov',action='store_true', default=False,
                    help=("if true, rbms are trained using nesterov momentum"))
parser.add_argument('--save',dest='save',action='store_true', default=False,
                    help="if true, the network is serialized and saved")
parser.add_argument('--train',dest='train',action='store_true', default=False,
                    help=("if true, the network is trained from scratch from the"
                          "traning data"))
parser.add_argument('--rbm', dest='rbm',action='store_true', default=False,
                    help=("if true, the code for traning an rbm on the data is run"))
parser.add_argument('--sparsity', dest='sparsity',action='store_true', default=False,
                    help=("if true, the the networks are trained with sparsity constraints"))
parser.add_argument('--dbKanade', dest='dbKanade',action='store_true', default=False,
                    help=("if true, the code for training a deepbelief net on the"
                          "data is run, where the supervised data is the Kanade DB"))
parser.add_argument('--dbPIE', dest='dbPIE',action='store_true', default=False,
                    help=("if true, the code for training a deepbelief net on the"
                          "data is run, where the supervised data is the PIE DB"))
parser.add_argument('--trainSize', type=int, default=10000,
                    help='the number of tranining cases to be considered')
parser.add_argument('--testSize', type=int, default=1000,
                    help='the number of testing cases to be considered')
parser.add_argument('netFile', help="file where the serialized network should be saved")
parser.add_argument('--nesterov', dest='nesterov',action='store_true', default=False,
                    help=("if true, the deep belief net is trained using nesterov momentum"))
parser.add_argument('--rmsprop', dest='rmsprop',action='store_true', default=False,
                    help=("if true, rmsprop is used when training the deep belief net."))
parser.add_argument('--rbmrmsprop', dest='rbmrmsprop',action='store_true', default=False,
                    help=("if true, rmsprop is used when training the rbms."))
parser.add_argument('--save_best_weights', dest='save_best_weights',action='store_true', default=False,
                    help=("if true, the best weights are used and saved during training."))
parser.add_argument('--cv', dest='cv',action='store_true', default=False,
                    help=("if true, do cross validation"))
parser.add_argument('--cvPIE', dest='cvPIE',action='store_true', default=False,
                    help=("if true, do cross validation"))
parser.add_argument('--illumination',dest='illumination',action='store_true', default=False,
                    help="if true, trains and tests the images with different illuminations")
parser.add_argument('--pose',dest='pose',action='store_true', default=False,
                    help="if true, trains and tests the images with different poses")
parser.add_argument('--subjects',dest='subjects',action='store_true', default=False,
                    help="if true, trains and tests the images with different subjects")
parser.add_argument('--missing', dest='missing',action='store_true', default=False,
                    help=("tests the network with missing data."))
parser.add_argument('--crossdb', dest='crossdb',action='store_true', default=False,
                    help=("if true, trains the DBN with multi pie and tests with Kanade."))
parser.add_argument('--crossdbCV', dest='crossdbCV',action='store_true', default=False,
                    help=("if true, trains the DBN with multi pie and tests with Kanade."))
parser.add_argument('--facedetection', dest='facedetection',action='store_true', default=False,
                    help=("if true, do face detection"))
parser.add_argument('--maxEpochs', type=int, default=1000,
                    help='the maximum number of supervised epochs')
parser.add_argument('--miniBatchSize', type=int, default=10,
                    help='the number of training points in a mini batch')
parser.add_argument('--validation',dest='validation',action='store_true', default=False,
                    help="if true, the network is trained using a validation set")
parser.add_argument('--equalize',dest='equalize',action='store_true', default=False,
                    help="if true, the input images are equalized before being fed into the net")
parser.add_argument('--crop',dest='crop',action='store_true', default=False,
                    help="crops images from databases before training the net")
parser.add_argument('--relu', dest='relu',action='store_true', default=False,
                    help=("if true, trains the RBM or DBN with a rectified linear unit"))
parser.add_argument('--preTrainEpochs', type=int, default=1,
                    help='the number of pretraining epochs')
parser.add_argument('--machine', type=int, default=0,
                    help='the host number of the machine running the experiment')
parser.add_argument('--kaggle',dest='kaggle',action='store_true', default=False,
                      help='if true, trains a net on the kaggle data')
parser.add_argument('--kagglecv',dest='kagglecv',action='store_true', default=False,
                      help='if true, cv for kaggle data')
parser.add_argument('--kagglesmall',dest='kagglesmall',action='store_true', default=False,
                      help='if true, cv for kaggle data')


# DEBUG mode?
parser.add_argument('--debug', dest='debug',action='store_false', default=False,
                    help=("if true, the deep belief net is ran in DEBUG mode"))

# Get the arguments of the program
args = parser.parse_args()

# Set the debug mode in the deep belief net
db.DEBUG = args.debug

SMALL_SIZE = ((40, 30))

def rbmEmotions(big=False, reconstructRandom=False):
  data, labels = readMultiPIE(big, equalize=args.equalize)
  print "data.shape"
  print data.shape

  if args.relu:
    activationFunction = Rectified()
    data = scale(data)
  else:
    activationFunction = Sigmoid()

  trainData = data[0:-1, :]
  # Train the network
  if args.train:
    # The number of hidden units is taken from a deep learning tutorial
    # The data are the values of the images have to be normalized before being
    # presented to the network
    nrVisible = len(data[0])
    nrHidden = 800
    # use 1 dropout to test the rbm for now
    net = rbm.RBM(nrVisible, nrHidden, 1.2, 1, 1,
                  visibleActivationFunction=activationFunction,
                  hiddenActivationFunction=activationFunction,
                  rmsprop=args.rbmrmsprop,
                  nesterov=args.rbmnesterov,
                  sparsityConstraint=args.sparsity,
                  sparsityRegularization=0.5,
                  trainingEpochs=args.maxEpochs,
                  sparsityTraget=0.01)
    net.train(trainData)
    t = visualizeWeights(net.weights.T, SMALL_SIZE, (10,10))
  else:
    # Take the saved network and use that for reconstructions
    f = open(args.netFile, "rb")
    t = pickle.load(f)
    net = pickle.load(f)
    f.close()

  # Show the initial image first
  test = data[-1, :]
  print "test.shape"
  print test.shape

  plt.imshow(vectorToImage(test, SMALL_SIZE), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('initialface.png', transparent=True)

  recon = net.reconstruct(test.reshape(1, test.shape[0]))
  print recon.shape

  plt.imshow(vectorToImage(recon, SMALL_SIZE), cmap=plt.cm.gray)
  plt.axis('off')
  plt.savefig('reconstructface.png', transparent=True)

  # Show the weights and their form in a tile fashion
  # Plot the weights
  plt.imshow(t, cmap=plt.cm.gray)
  plt.axis('off')
  if args.rbmrmsprop:
    st='rmsprop'
  else:
    st = 'simple'
  plt.savefig('weights' + st + '.png', transparent=True)

  # let's make some sparsity checks
  hidden = net.hiddenRepresentation(test.reshape(1, test.shape[0]))
  print hidden.sum()
  print "done"

  if args.save:
    f = open(args.netFile, "wb")
    pickle.dump(t, f)
    pickle.dump(net, f)


def deepbeliefKanadeCV(big=False):
  """
    Arguments:
      big: should the big or small images be used?
  """
  data, labels = readKanade(big, None, equalize=args.equalize, train=True)

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  if args.relu:
    activationFunction = Rectified()
    rbmActivationFunctionVisible = Identity()
    rbmActivationFunctionHidden = RectifiedNoisy()
    data = scale(data)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()

  params =[(0.001, 0.01, 0.9),  (0.001, 0.05, 0.9),  (0.005, 0.01, 0.9),  (0.005, 0.05, 0.9),
           (0.001, 0.01, 0.95), (0.001, 0.05, 0.95), (0.005, 0.01, 0.95), (0.005, 0.05, 0.95),
           (0.001, 0.01, 0.99), (0.001, 0.05, 0.99), (0.005, 0.01, 0.99), (0.005, 0.05, 0.99)]

  unsupervisedData = buildUnsupervisedDataSetForKanadeLabelled()

  kf = cross_validation.KFold(n=len(data), k=len(params))
  bestCorrect = 0
  bestProbs = 0

  fold = 0
  for train, test in kf:
    trainData = data[train]
    trainLabels = labels[train]

    net = db.DBN(5, [1200, 1800, 1800, 1800, 7],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=params[fold][0],
               supervisedLearningRate=params[fold][1],
               momentumMax=params[fold][2],
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               save_best_weights=args.save_best_weights,
               hiddenDropout=0.5,
               momentumFactorForLearningRateRBM=False,
               visibleDropout=0.8,
               rbmHiddenDropout=1.0,
               rbmVisibleDropout=1.0,
               preTrainEpochs=args.preTrainEpochs)

    net.train(trainData, trainLabels,
              maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData)

    probs, predicted = net.classify(data[test])

    testLabels = labels[test]
    correct = 0
    errorCases = []

    for i in xrange(len(test)):
      print "predicted"
      print "probs"
      print probs[i]
      print predicted[i]
      print "actual"
      actual = testLabels[i]
      print np.argmax(actual)
      if predicted[i] == np.argmax(actual):
        correct += 1
      else:
        errorCases.append(i)

    print "correct for " + str(params[fold])
    print correct

    if bestCorrect < correct:
      bestCorrect = correct
      bestParam = params[fold]
      bestProbs = correct * 1.0 / len(test)

    fold += 1

  print "bestParam"
  print bestParam

  print "bestProbs"
  print bestProbs


def deepbeliefKanade(big=False):
  trainData, trainLabels = readKanade(big, None, equalize=args.equalize, train=True)
  testData, testLabels = readKanade(big, None, equalize=args.equalize, train=False)

  unsupervisedData = buildUnsupervisedDataSetForKanadeLabelled()

  if args.relu:
    activationFunction = Rectified()
    data = scale(data)
    rbmActivationFunctionVisible = Identity()
    rbmActivationFunctionHidden = RectifiedNoisy()

    if unsupervisedData == None:
      unsupervisedLearningRate = 0.005
      supervisedLearningRate = 0.01
      momentumMax = 0.99
    else:
      unsupervisedLearningRate = 0.001
      supervisedLearningRate = 0.01
      momentumMax = 0.99
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()

    unsupervisedLearningRate = 0.5
    supervisedLearningRate = 0.1
    momentumMax = 0.9

  net = db.DBN(5, [1200, 1500, 1500, 1500, 7],
             binary=1-args.relu,
             activationFunction=activationFunction,
             rbmActivationFunctionVisible=rbmActivationFunctionVisible,
             rbmActivationFunctionHidden=rbmActivationFunctionHidden,
             unsupervisedLearningRate=unsupervisedLearningRate,
             supervisedLearningRate=supervisedLearningRate,
             momentumMax=momentumMax,
             nesterovMomentum=args.nesterov,
             rbmNesterovMomentum=args.rbmnesterov,
             rmsprop=args.rmsprop,
             save_best_weights=args.save_best_weights,
             miniBatchSize=args.miniBatchSize,
             hiddenDropout=0.5,
             visibleDropout=0.8,
             momentumFactorForLearningRateRBM=False,
             firstRBMheuristic=False,
             rbmVisibleDropout=1.0,
             rbmHiddenDropout=1.0,
             preTrainEpochs=args.preTrainEpochs,
             sparsityConstraintRbm=args.sparsity,
             sparsityRegularizationRbm=0.001,
             sparsityTragetRbm=0.01)

  net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
            validation=args.validation,
            unsupervisedData=unsupervisedData)

  probs, predicted = net.classify(testData)

  correct = 0
  errorCases = []

  for i in xrange(len(test)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = testLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(test)

  confMatrix = confusion_matrix(np.argmax(testLabels, axis=1), predicted)
  print "confusion matrix"
  print confMatrix

  print classification_report(np.argmax(testLabels, axis=1), predicted)

  if args.save:
    with open(args.netFile, "wb") as f:
      pickle.dump(net, f)


def buildUnsupervisedDataSetForKanadeLabelled():
  return np.vstack((readJaffe(args.crop, args.facedetection, equalize=args.equalize),
                    readNottingham(args.crop, args.facedetection, equalize=args.equalize)))

def buildUnsupervisedDataSetForPIE():
  return None


def deepbeliefMultiPIE(big=False):
  trainData, trainLabels = readMultiPIE(equalize=args.equalize, train=True)
  testData, testLabels = readMultiPIE(equalize=args.equalize, train=True)

  if args.relu:
    activationFunction = RectifiedNoisy()
    rbmActivationFunctionHidden = RectifiedNoisy()
    rbmActivationFunctionVisible = Identity()
    unsupervisedLearningRate = 0.005
    supervisedLearningRate = 0.001
    momentumMax = 0.95
    data = scale(data)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95

  if args.train:
    net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=unsupervisedLearningRate,
               supervisedLearningRate=supervisedLearningRate,
               momentumMax=momentumMax,
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               save_best_weights=args.save_best_weights,
               miniBatchSize=args.miniBatchSize,
               visibleDropout=0.8,
               hiddenDropout=0.5,
               rbmHiddenDropout=1.0,
               rbmVisibleDropout=1.0,
               preTrainEpochs=args.preTrainEpochs)

    unsupervisedData = buildUnsupervisedDataSetForPIE()

    net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData)
  else:
    # Take the saved network and use that for reconstructions
    with open(args.netFile, "rb") as f:
      net = pickle.load(f)

  probs, predicted = net.classify(testData)

  correct = 0
  errorCases = []

  for i in xrange(len(test)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = testLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(test)

  print type(predicted)
  print type(testLabels)
  print predicted.shape
  print testLabels.shape

  confMatrix = confusion_matrix(np.argmax(testLabels, axis=1), predicted)

  print "confusion matrix"
  print confMatrix

  if args.save:
    with open(args.netFile, "wb") as f:
      pickle.dump(net, f)


def deepbeliefPIECV(big=False):
  data, labels = readMultiPIE(equalize=args.equalize, train=True)
  data, labels = shuffle(data, labels)

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  if args.relu:
    activationFunction = Rectified()
    rbmActivationFunctionVisible = Identity()
    rbmActivationFunctionHidden = RectifiedNoisy()
    # IMPORTANT: SCALE THE DATA IF YOU USE GAUSSIAN VISIBlE UNITS
    data = scale(data)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()

  if args.relu:
    params = [(0.005, 0.001, 0.95, 0.8, 1.0), (0.005, 0.001, 0.95, 1.0, 1.0), (0.005, 0.001, 0.95, 0.8, 0.5), (0.05, 0.01, 0.95, 1.0, 0.5)]
  else:
    params =[(0.05, 0.01, 0.95,  0.8, 1.0), (0.05, 0.01, 0.95, 1.0, 1.0), (0.05, 0.01, 0.95,  0.8, 0.5), (0.05, 0.01, 0.95, 1.0, 0.5)]

  print "cv for params"
  print params
  unsupervisedData = buildUnsupervisedDataSetForPIE()

  kf = cross_validation.KFold(n=len(data), k=len(params))
  bestCorrect = 0
  bestProbs = 0

  probsforParms = []
  fold = 0
  for train, test in kf:

    trainData = data[train]
    trainLabels = labels[train]

    net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=params[fold][0],
               supervisedLearningRate=params[fold][1],
               momentumMax=params[fold][2],
               nesterovMomentum=args.nesterov,
               save_best_weights=args.save_best_weights,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               rbmHiddenDropout=1.0,
               rbmVisibleDropout=1.0,
               visibleDropout=params[fold][3],
               hiddenDropout=params[fold][4],
               preTrainEpochs=args.preTrainEpochs)

    net.train(trainData, trainLabels,
              maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData)

    probs, predicted = net.classify(data[test])

    testLabels = labels[test]
    correct = 0
    errorCases = []

    for i in xrange(len(test)):
      print "predicted"
      print "probs"
      print probs[i]
      print predicted[i]
      print "actual"
      actual = testLabels[i]
      print np.argmax(actual)
      if predicted[i] == np.argmax(actual):
        correct += 1
      else:
        errorCases.append(i)

    print "correct for " + str(params[fold])
    print correct
    print "correctProbs"
    correctProbs = correct * 1.0 / len(test)
    print correctProbs

    probsforParms += [correctProbs]

    if bestCorrect < correct:
      bestCorrect = correct
      bestParam = params[fold]
      bestProbs = correctProbs

    with open("resultsPIECV" +args.relu+".txt", "a") as resfile:
      resfile.write(str(params[fold]))
      resfile.write(str(correctProbs))
      resfile.write(str(correct))

    fold += 1

  print "bestParam"
  print bestParam

  print "bestProbs"
  print bestProbs


  for i in xrange(len(params)):
    print "parameter tuple " + str(params[i]) + " achieved correctness of " + str(probsforParms[i])

  with open("resultsPIECV.txt", "a") as resfile:
    for i in xrange(len(params)):
      test = "parameter tuple " + str(params[i]) + " achieved correctness of " + str(probsforParms[i])
      resfile.write(str(params[i]))

def deepbeliefKaggleCompetitionSmallDataset(big=False):
  print "you are using the net file" , args.netFile
  trainData, trainLabels = readKaggleCompetitionSmallDataset(args.equalize, train=True)
  testData, testLabels = readKaggleCompetitionSmallDataset(args.equalize, train=False)

  if args.relu:
    activationFunction = Rectified()
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95
    data = scale(data)
    rbmActivationFunctionVisible = Identity()
    rbmActivationFunctionHidden = RectifiedNoisy()
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()

    unsupervisedLearningRate = 0.5
    supervisedLearningRate = 0.1
    momentumMax = 0.9

  if args.train:
    net = db.DBN(5, [2304, 1500, 1500, 1500, 7],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=unsupervisedLearningRate,
               supervisedLearningRate=supervisedLearningRate,
               momentumMax=momentumMax,
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               save_best_weights=args.save_best_weights,
               firstRBMheuristic=False,
               hiddenDropout=0.5,
               visibleDropout=0.8,
               rbmVisibleDropout=1.0,
               rbmHiddenDropout=1.0,
               initialInputShape=(48, 48),
               preTrainEpochs=args.preTrainEpochs)

    # unsupervisedData = readKaggleCompetitionUnlabelled()
    unsupervisedData = None

    net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData)
  else:
    # Take the saved network and use that for reconstructions
    with open(args.netFile, "rb") as f:
      net = pickle.load(f)

  print "nr layers: ", net.layerSizes

  probs, predicted = net.classify(testData)

  correct = 0
  errorCases = []

  for i in xrange(len(test)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = testLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(test)

  confMatrix = confusion_matrix(np.argmax(testLabels, axis=1), predicted)
  print "confusion matrix"
  print confMatrix

  if args.save:
    with open(args.netFile, "wb") as f:
      print "you are saving in file", args.netFile
      pickle.dump(net, f)

def deepbeliefKaggleCompetition(big=False):
  trainData, trainLabels = readBigKaggleTrain()

  trainData, trainLabels = shuffle(trainData, trainLabels)

  trainData = trainData[0:args.trainSize]
  trainLabels = trainLabels[0:args.trainSize]

  testData, testLabels = readBigKaggleTestPublic()


  testData = testData[0:args.trainSize]
  testLabels = testLabels[0:args.trainSize]

  if args.relu:
    activationFunction = Rectified()
    unsupervisedLearningRate = 0.001
    supervisedLearningRate = 0.001
    momentumMax = 0.95
    trainData = scale(trainData)
    trainData = scale(trainData)
    rbmActivationFunctionVisible = Identity()
    rbmActivationFunctionHidden = RectifiedNoisy()

  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()

    unsupervisedLearningRate = 0.5
    supervisedLearningRate = 0.1
    momentumMax = 0.9

  net = db.DBN(5, [2304, 1500, 1500, 2000, 7],
             binary=1-args.relu,
             activationFunction=activationFunction,
             rbmActivationFunctionVisible=rbmActivationFunctionVisible,
             rbmActivationFunctionHidden=rbmActivationFunctionHidden,
             unsupervisedLearningRate=unsupervisedLearningRate,
             supervisedLearningRate=supervisedLearningRate,
             momentumMax=momentumMax,
             nesterovMomentum=args.nesterov,
             rbmNesterovMomentum=args.rbmnesterov,
             rmsprop=args.rmsprop,
             miniBatchSize=args.miniBatchSize,
             save_best_weights=args.save_best_weights,
             firstRBMheuristic=False,
             hiddenDropout=0.5,
             visibleDropout=0.8,
             rbmVisibleDropout=1.0,
             rbmHiddenDropout=1.0,
             preTrainEpochs=args.preTrainEpochs)

  # unsupervisedData = readKaggleCompetitionUnlabelled()
  unsupervisedData = None

  net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
            validation=args.validation,
            unsupervisedData=unsupervisedData)

  probs, predicted = net.classify(testData)

  correct = 0
  errorCases = []

  for i in xrange(len(testLabels)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = testLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(testLabels)

  confMatrix = confusion_matrix(np.argmax(testLabels, axis=1), predicted)
  print "confusion matrix"
  print confMatrix

  if args.save:
    with open(args.netFile, "wb") as f:
      pickle.dump(net, f)

def deepbeliefKaggleCompetitionBigCV():
  data, labels = readBigKaggleTrain()
  data, labels = shuffle(data, labels)

  data = data[0:args.trainSize]
  labels = labels[0:args.trainSize]

  assert args.relu, " only rectified linear units are supported for second kaggle competition"

  activationFunction = Rectified()
  momentumMax = 0.95
  data = scale(data)
  rbmActivationFunctionVisible = Identity()
  rbmActivationFunctionHidden = RectifiedNoisy()

  if args.machine == 0:
    params = [(0.005, 0.005),]
  elif args.machine == 1:
    params = [(0.001, 0.001)]
  elif args.machine == 2:
    params = [(0.001, 0.005)]
  elif args.machine == 3:
    params = [(0.005, 0.001)]

  kf = cross_validation.KFold(n=len(data), k=5)

  bestCorrect = 0
  bestProbs = 0
  probsforParms = []
  fold = 0
  for train, test in kf:
    if fold >= len(params):
      break
    trainData = data[train]
    trainLabels = labels[train]

    testData = data[test]
    testLabels = labels[test]

    net = db.DBN(5, [2304, 1500, 1500, 1500, 7],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=params[fold][0],
               supervisedLearningRate=params[fold][1],
               momentumMax=momentumMax,
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               save_best_weights=args.save_best_weights,
               firstRBMheuristic=False,
               hiddenDropout=0.5,
               visibleDropout=0.8,
               rbmVisibleDropout=1.0,
               rbmHiddenDropout=1.0,
               preTrainEpochs=args.preTrainEpochs)

    unsupervisedData = None

    net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData)

    probs, predicted = net.classify(testData)

    correct = 0
    errorCases = []

    for i in xrange(len(testLabels)):
      actual = testLabels[i]
      print np.argmax(actual)
      if predicted[i] == np.argmax(actual):
        correct += 1

    correctProbs = correct * 1.0 / len(test)
    probsforParms += [correctProbs]

    if bestCorrect < correct:
      bestCorrect = correct
      bestParam = params[fold]
      bestProbs = correctProbs

    with open("resultsKagglemachine" + str(args.machine) + ".txt", "a") as resfile:
      resfile.write(str(params[fold]))
      resfile.write(str(correctProbs))
      resfile.write(str(correct))

    print "correct"
    print correct
    print "percentage correct"
    print correct  * 1.0/ len(testLabels)

    confMatrix = confusion_matrix(np.argmax(testLabels, axis=1), predicted)
    print "confusion matrix"
    print confMatrix

    fold += 1

  for i in xrange(len(params)):
    print "parameter tuple " + str(params[i]) + " achieved correctness of " + str(probsforParms[i])

  with open("resultsKagglemachine" + str(args.machine)+ ".txt", "a") as resfile:
    for i in xrange(len(params)):
      test = "parameter tuple " + str(params[i]) + " achieved correctness of " + str(probsforParms[i])
      resfile.write(str(params[i]))


def deepBeliefPieDifferentConditions():
  if args.illumination:
    getDataFunction = readMultiPieDifferentIlluminations
    allConditions = np.array(range(5))
  elif args.pose:
    getDataFunction = readMultiPieDifferentPoses
    allConditions = np.array(range(5))
  elif args.subjects:
    getDataFunction = readMultiPieDifferentSubjects
    allConditions = np.array(range(147))

  kf = cross_validation.KFold(n=len(allConditions), k=5)

  confustionMatrices = []
  correctAll = []

  for trainConditions, _ in kf:
    print "trainConditions"
    print trainConditions
    print trainConditions.shape

    trainData, trainLabels, testData, testLabels = getDataFunction(trainConditions, equalize=args.equalize)

    trainData, trainLabels = shuffle(trainData, trainLabels)

    print "input shape"
    print trainData[0].shape
    print "type(trainData)"
    print type(trainData)

    if args.relu:
      activationFunction = Rectified()
      rbmActivationFunctionVisible = Identity()
      rbmActivationFunctionHidden = RectifiedNoisy()

      unsupervisedLearningRate = 0.005
      supervisedLearningRate = 0.001
      momentumMax = 0.95
      trainData = scale(trainData)
      testData = scale(testData)

    else:
      activationFunction = Sigmoid()
      rbmActivationFunctionVisible = Sigmoid()
      rbmActivationFunctionHidden = Sigmoid()

      unsupervisedLearningRate = 0.05
      supervisedLearningRate = 0.01
      momentumMax = 0.9

    if args.train:
      net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
                 binary=1-args.relu,
                 activationFunction=activationFunction,
                 rbmActivationFunctionVisible=rbmActivationFunctionVisible,
                 rbmActivationFunctionHidden=rbmActivationFunctionHidden,
                 unsupervisedLearningRate=unsupervisedLearningRate,
                 # is this not a bad learning rate?
                 supervisedLearningRate=supervisedLearningRate,
                 momentumMax=momentumMax,
                 nesterovMomentum=args.nesterov,
                 rbmNesterovMomentum=args.rbmnesterov,
                 rmsprop=args.rmsprop,
                 miniBatchSize=args.miniBatchSize,
                 save_best_weights=args.save_best_weights,
                 visibleDropout=0.8,
                 hiddenDropout=1.0,
                 rbmHiddenDropout=1.0,
                 rbmVisibleDropout=1.0,
                 preTrainEpochs=args.preTrainEpochs)

      print "trainData.shape"
      print trainData.shape
      net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
                validation=args.validation,
                unsupervisedData=None)
    else:
      # Take the saved network and use that for reconstructions
      with open(args.netFile, "rb") as f:
        net = pickle.load(f)

    probs, predicted = net.classify(testData)

    correct = 0
    errorCases = []

    for i in xrange(len(testLabels)):
      print "predicted"
      print "probs"
      print probs[i]
      print "predicted"
      print predicted[i]
      print "actual"
      actual = testLabels[i]
      print np.argmax(actual)
      if predicted[i] == np.argmax(actual):
        correct += 1
      else:
        errorCases.append(i)

    print "correct"
    print correct

    print "percentage correct"
    correct = correct  * 1.0/ len(testLabels)
    print correct

    confMatrix = confusion_matrix(np.argmax(testLabels, axis=1), predicted)

    print "confusion matrix"
    print confMatrix

    confustionMatrices += [confMatrix]
    correctAll += [correct]


  for i in allConditions:
    print "for condition" + str(i)
    print "the correct rate was " + str(correctAll[i])
    print "the confusionMatrix was " + str(confustionMatrices[i])

  print "average correct rate", sum(correctAll) * 1.0 / len(correctAll)
  print "average confusionMatrix was ", sum(confustionMatrices) * 1.0 / len(confustionMatrices)


"""Train with PIE test with Kanade. Check the equalization code. """
def crossDataBase():
  # Only train with the frontal pose
  trainData, trainLabels, _, _ = readMultiPieDifferentPoses([2], equalize=args.equalize)
  trainData, trainLabels = shuffle(trainData, trainLabels)

  print "trainLabels"
  print np.argmax(trainLabels, axis=1)

  testData, testLabels = readKanade(
      False, None, equalize=args.equalize, vectorizeLabels=False, train=False)
  print "testLabels"
  print testLabels
  # Some emotions do not correspond for a to b, so we have to map them
  testData, testLabels = mapKanadeToPIELabels(testData, testLabels)
  testLabels = labelsToVectors(testLabels, 6)

  labelsSimple = np.argmax(testLabels, axis=1)

  if args.relu:
    activationFunction = Rectified()
    rbmActivationFunctionHidden = RectifiedNoisy()
    rbmActivationFunctionVisible = Identity()

    unsupervisedLearningRate = 0.005
    supervisedLearningRate = 0.001
    momentumMax = 0.95
    trainData = scale(trainData)
    testData = scale(testData)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95

  if args.train:
    net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=unsupervisedLearningRate,
               supervisedLearningRate=supervisedLearningRate,
               momentumMax=momentumMax,
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               save_best_weights=args.save_best_weights,
               visibleDropout=0.8,
               hiddenDropout=1.0,
               rbmHiddenDropout=1.0,
               rbmVisibleDropout=1.0,
               preTrainEpochs=args.preTrainEpochs)

    unsupervisedData = buildUnsupervisedDataSetForPIE()

    net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData)

    if args.save:
      with open(args.netFile, "wb") as f:
        pickle.dump(net, f)

  else:
     # Take the saved network and use that for reconstructions
    with open(args.netFile, "rb") as f:
      net = pickle.load(f)

  probs, predicted = net.classify(testData)

  correct = 0
  errorCases = []

  for i in xrange(len(testLabels)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = testLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(testLabels)

  confMatrix = confusion_matrix(np.argmax(testLabels, axis=1), predicted)

  print "confusion matrix"
  print confMatrix

"""Train with PIE test with Kanade. Check the equalization code. """
def crossDataBaseCV():
  # Only train with the frontal pose
  trainData, trainLabels, _, _ = readMultiPieDifferentPoses([2], equalize=args.equalize)
  trainData, trainLabels = shuffle(trainData, trainLabels)

  print "trainLabels"
  print np.argmax(trainLabels, axis=1)

  confustionMatrices = []
  correctAll = []

  params = [(0.005, 0.005), (0.001, 0.005), (0.001, 0.05), (0.01, 0.05), (0.01, 0.005)]

  # The test data to be used during CV should be the taken from the training set, so
  # that at test time we do not test with the same data used for finding hyperparameters
  testData, testLabels = readKanade(
      False, None, equalize=args.equalize, vectorizeLabels=False, train=True)
  print "testLabels"
  print testLabels
  # Some emotions do not correspond for a to b, so we have to map them
  testData, testLabels = mapKanadeToPIELabels(testData, testLabels)
  testLabels = labelsToVectors(testLabels, 6)

  print "testLabels after map"
  labelsSimple = np.argmax(testLabels, axis=1)
  print labelsSimple

  if args.relu:
    activationFunction = Rectified()
    rbmActivationFunctionHidden = RectifiedNoisy()
    rbmActivationFunctionVisible = Identity()
    trainData = scale(trainData)
    testData = scale(testData)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()

  for param in params:
    if args.train:
      net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
                 binary=1-args.relu,
                 activationFunction=activationFunction,
                 rbmActivationFunctionVisible=rbmActivationFunctionVisible,
                 rbmActivationFunctionHidden=rbmActivationFunctionHidden,
                 supervisedLearningRate=param[1],
                 unsupervisedLearningRate=param[0],
                 momentumMax=0.95,
                 nesterovMomentum=args.nesterov,
                 rbmNesterovMomentum=args.rbmnesterov,
                 rmsprop=args.rmsprop,
                 miniBatchSize=args.miniBatchSize,
                 save_best_weights=args.save_best_weights,
                 visibleDropout=0.8,
                 hiddenDropout=1.0,
                 rbmHiddenDropout=1.0,
                 rbmVisibleDropout=1.0,
                 preTrainEpochs=args.preTrainEpochs)

      unsupervisedData= buildUnsupervisedDataSetForPIE()

      net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
                validation=args.validation,
                unsupervisedData=unsupervisedData)

    else:
       # Take the saved network and use that for reconstructions
      with open(args.netFile, "rb") as f:
        net = pickle.load(f)

    probs, predicted = net.classify(testData)

    correct = 0
    errorCases = []

    for i in xrange(len(testLabels)):
      print "predicted"
      print "probs"
      print probs[i]
      print "predicted"
      print predicted[i]
      print "actual"
      actual = testLabels[i]
      print np.argmax(actual)
      if predicted[i] == np.argmax(actual):
        correct += 1
      else:
        errorCases.append(i)

    print "correct"
    print correct

    print "percentage correct"
    print correct  * 1.0/ len(testLabels)

    confMatrix = confusion_matrix(np.argmax(testLabels, axis=1), predicted)

    correctAll += [correct  * 1.0/ len(testLabels)]
    confustionMatrices += [confMatrix]

    print "confusion matrix"
    print confMatrix

  for i, param in enumerate(params):
    print "for param" + str(param)
    print "the correct rate was " + str(correctAll[i])
    print "the confusionMatrix was " + str(confustionMatrices[i])


def addBlobsOfMissingData(testData, sqSize=5, returnIndices=False):
  maxHeight = SMALL_SIZE[0] - sqSize
  maxLength = SMALL_SIZE[1] - sqSize

  def makeBlob(x):
    x = x.reshape(SMALL_SIZE)
    m = np.random.random_integers(low=0, high=maxHeight)
    n = np.random.random_integers(low=0, high=maxLength)

    for i in xrange(sqSize):
      for j in xrange(sqSize):
        x[m + i, n + j] = 0

    if returnIndices:
      return x.reshape(-1), (m, n)
    else:
      return x.reshape(-1)

  if returnIndices:
    imgsAndIndices = map(makeBlob, testData)
    imgs = [x[0] for x in imgsAndIndices]
    indices = [x[1] for x in imgsAndIndices]
    return np.array(imgs), indices

  else:
    return np.array(map(makeBlob, testData))


def makeMissingDataImage():
  data, labels = readMultiPIE(equalize=args.equalize)
  data, labels = shuffle(data, labels)

  testData = data[0:20]
  testData = addBlobsOfMissingData(testData, sqSize=10)

  final = []
  for i in xrange(6):
    final += [testData[i].reshape(SMALL_SIZE)]

  final = np.hstack(tuple(final))

  plt.imshow(final, cmap=plt.cm.gray, interpolation="nearest")
  plt.axis('off')
  plt.show()


def getVariance(middle):
  eps = 0.001
  return  np.sqrt(- middle * middle / np.log(eps))

# Required for the missing data plot
def makeGaussianRect(n):
  middle = (n - 1) * 1.0 / 2
  # middle = n / 2

  var = getVariance(n) # TODO: compute this
  # var = 1 # TODO: compute this
  def gaussianDistance(x, y):
    return np.exp(- ((x - middle) **2 + (y-middle) **2) / (2 * var))

  res = np.zeros((n,n))
  for i in xrange(n):
    for j in xrange(n):
      res[i,j] = gaussianDistance(i,j)

  return res


"""Train with PIE test with Kanade. Check the equalization code. """
def missingData():
  trainData, trainLabels = readMultiPIE(equalize=args.equalize, train=True)
  testData, testLabels = readMultiPIE(equalize=args.equalize, train=False)
  squaresize = 10

  allTestData  = []
  allIndices  = []
  allTestLabels = []

  for i in xrange(5):
    testDataMissing, indicesMissing = addBlobsOfMissingData(testData, sqSize=squaresize, returnIndices=True)
    allTestData += [testDataMissing]
    allIndices += [indicesMissing]
    allTestLabels += [testLabels]

  testData = np.vstack(allTestData)
  indices = np.vstack(allIndices)
  testLabels  = np.vstack(allTestLabels)

  print "testLabels.shape"
  print testLabels.shape

  gaussDistances = makeGaussianRect(10)

  for i in xrange(3):
    plt.imshow(vectorToImage(testData[i], SMALL_SIZE), cmap=plt.cm.gray, interpolation="nearest")
    plt.show()

  if args.relu:
    activationFunction = Rectified()
    rbmActivationFunctionHidden = RectifiedNoisy()
    rbmActivationFunctionVisible = Identity()
    unsupervisedLearningRate = 0.005
    supervisedLearningRate = 0.001
    momentumMax = 0.95
    trainData = scale(trainData)
    testData = scale(testData)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95

  if args.train:
    net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=unsupervisedLearningRate,
               supervisedLearningRate=supervisedLearningRate,
               momentumMax=momentumMax,
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               save_best_weights=args.save_best_weights,
               miniBatchSize=args.miniBatchSize,
               visibleDropout=0.8,
               hiddenDropout=1.0,
               rbmHiddenDropout=1.0,
               rbmVisibleDropout=1.0,
               preTrainEpochs=args.preTrainEpochs)

    unsupervisedData = buildUnsupervisedDataSetForPIE()

    net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData,
              trainingIndices=train)

    if args.save:
      with open(args.netFile, "wb") as f:
        pickle.dump(net, f)
  else:
    # Take the saved network and use that for reconstructions
    print "using ", args.netFile, " for reading the pickled net"
    with open(args.netFile, "rb") as f:
      net = pickle.load(f)

      trainingIndices = net.trainingIndices
      testIndices = np.setdiff1d(np.arange(len(data)), trainingIndices)
      testData = data[testIndices]
      print "len(testData)"
      print len(testData)
      testData = addBlobsOfMissingData(testData, sqSize=5)


  total = np.zeros((40, 30))
  errorDict = np.zeros((40, 30))

  probs, predicted = net.classify(testData)

  correct = 0
  errorCases = []

  for i in xrange(len(testLabels)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = testLabels[i]
    print np.argmax(actual)

    m, n = indices[i]
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      for i in xrange(squaresize):
        for j in xrange(squaresize):
          errorDict[m + i, n + j] += gaussDistances[i,j]
      errorCases.append(i)

    for i in xrange(squaresize):
      for j in xrange(squaresize):
        total[m + i, n + j] += 1

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(testLabels)

  confMatrix = confusion_matrix(np.argmax(testLabels, axis=1), predicted)

  print "confusion matrix"
  print confMatrix

  with open("missingDatamat", "wb") as f:
        pickle.dump(total, f)
        pickle.dump(errorDict, f)

  print total
  plt.matshow(total, cmap=plt.get_cmap("YlOrRd"),interpolation='none')
  plt.show()


def makeMissingDataOnly12Positions(testData):
  def makeBlob(x):
    x = x.reshape(SMALL_SIZE)
    m = np.random.randint(low=0, high=4)
    n = np.random.randint(low=0, high=3)

    for i in xrange(10):
      for j in xrange(10):
        x[10 * m  + i, 10 * n + j] = 0

    return x.reshape(-1), (m,n)

  data = []
  coordinates = []
  for i, d in enumerate(testData):
    d, (m, n) = makeBlob(d)
    data += [d]
    coordinates += [(m,n)]

  return np.array(data), coordinates

def missingDataTestFromTrainedNet():
  trainData, trainLabels = readMultiPIE(equalize=args.equalize, train=True)
  testData, testLabels = readMultiPIE(equalize=args.equalize, train=False)

  testData, pairs = makeMissingDataOnly12Positions(testData)

  if args.relu:
    activationFunction = Rectified()
    rbmActivationFunctionHidden = RectifiedNoisy()
    rbmActivationFunctionVisible = Identity()
    unsupervisedLearningRate = 0.005
    supervisedLearningRate = 0.001
    momentumMax = 0.95
    trainData = scale(trainData)
    testData = scale(testData)
  else:
    activationFunction = Sigmoid()
    rbmActivationFunctionHidden = Sigmoid()
    rbmActivationFunctionVisible = Sigmoid()
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95

  if args.train:
    net = db.DBN(5, [1200, 1500, 1500, 1500, 6],
               binary=1-args.relu,
               activationFunction=activationFunction,
               rbmActivationFunctionVisible=rbmActivationFunctionVisible,
               rbmActivationFunctionHidden=rbmActivationFunctionHidden,
               unsupervisedLearningRate=unsupervisedLearningRate,
               supervisedLearningRate=supervisedLearningRate,
               momentumMax=momentumMax,
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               save_best_weights=args.save_best_weights,
               visibleDropout=0.8,
               hiddenDropout=1.0,
               rbmHiddenDropout=1.0,
               rbmVisibleDropout=1.0,
               preTrainEpochs=args.preTrainEpochs)

    unsupervisedData = buildUnsupervisedDataSetForPIE()

    net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData,
              trainingIndices=train)

    if args.save:
      with open(args.netFile, "wb") as f:
        pickle.dump(net, f)

  dictSquares = {}
  for i in xrange(4):
    for j in xrange(3):
      dictSquares[(i,j)] = []

  probs, predicted = net.classify(testData)
  correct = 0
  errorCases = []

  for i in xrange(len(testLabels)):
    print "predicted"
    print "probs"
    print probs[i]
    print "predicted"
    print predicted[i]
    print "actual"
    actual = testLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
      dictSquares[pairs[i]] += [1]
    else:
      errorCases.append(i)
      dictSquares[pairs[i]] += [0]

  print "percentage correct"
  print correct  * 1.0/ len(testLabels)

  mat = np.zeros((4, 3))
  for i in xrange(4):
    for j in xrange(3):
      print "len(dictSquares[(i,j)])"
      print len(dictSquares[(i,j)])
      mat[i,j] = sum(dictSquares[(i,j)]) * 1.0 / len(dictSquares[(i,j)])


  print mat
  plt.matshow(mat, cmap=plt.get_cmap("YlOrRd"),interpolation='none')
  plt.show()


def main():
  if args.rbm:
    rbmEmotions()
  if args.cv:
    deepbeliefKanadeCV()
  if args.dbKanade:
    deepbeliefKanade()
  if args.dbPIE:
    deepbeliefMultiPIE()
  if args.cvPIE:
    deepbeliefPIECV()
  if args.crossdb:
    crossDataBase()
  if args.crossdbCV:
    crossDataBaseCV()

  if args.illumination or args.pose or args.subjects:
    deepBeliefPieDifferentConditions()

  if args.missing:
    missingData()

  if args.kaggle:
    deepbeliefKaggleCompetition()

  if args.kagglecv:
    deepbeliefKaggleCompetitionBigCV()

  if args.kagglesmall:
    deepbeliefKaggleCompetitionSmallDataset()


if __name__ == '__main__':
  import random
  print "FIXING RANDOMNESS"
  random.seed(6)
  np.random.seed(6)
  main()
