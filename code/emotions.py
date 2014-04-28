""" The aim of this file is to contain all the function
and the main which have to do with emotion recognition, especially
with the Kanade database."""

import glob
import argparse
# import DimensionalityReduction
import cPickle as pickle
from sklearn import cross_validation

import os
import fnmatch
import matplotlib.image as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage import io
from skimage import color
import os
import cv2

import facedetection


import deepbelief as db
import restrictedBoltzmannMachine as rbm
from common import *

parser = argparse.ArgumentParser(description='digit recognition')
parser.add_argument('--rbmnesterov', dest='rbmnesterov',action='store_true', default=False,
                    help=("if true, rbms are trained using nesterov momentum"))
parser.add_argument('--save',dest='save',action='store_true', default=False,
                    help="if true, the network is serialized and saved")
parser.add_argument('--train',dest='train',action='store_true', default=False,
                    help=("if true, the network is trained from scratch from the"
                          "traning data"))
parser.add_argument('--rbm', dest='rbm',action='store_true', default=False,
                    help=("if true, the code for traning an rbm on the data is run"))
parser.add_argument('--db', dest='db',action='store_true', default=False,
                    help=("if true, the code for traning a deepbelief net on the"
                          "data is run"))
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
parser.add_argument('--cv', dest='cv',action='store_true', default=False,
                    help=("if true, do cross validation"))
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
parser.add_argument('--relu', dest='relu',action='store_true', default=False,
                    help=("if true, trains the RBM or DBN with a rectified linear unit"))

# DEBUG mode?
parser.add_argument('--debug', dest='debug',action='store_false', default=False,
                    help=("if true, the deep belief net is ran in DEBUG mode"))

# Get the arguments of the program
args = parser.parse_args()

# Set the debug mode in the deep belief net
db.DEBUG = args.debug

SMALL_SIZE = ((40, 30))


def equalize(x):
  y = x.reshape(SMALL_SIZE)
  return cv2.equalizeHist(y).reshape(-1)

def rbmEmotions(big=False, reconstructRandom=False):
  data, labels = readKanade(big)
  print "data.shape"
  print data.shape

  trainData = data[0:-1, :]

  if args.relu:
    activationFunction = relu
  else:
    activationFunction = T.nnet.sigmoid

  # Train the network
  if args.train:
    # The number of hidden units is taken from a deep learning tutorial
    # The data are the values of the images have to be normalized before being
    # presented to the network
    nrVisible = len(data[0])
    nrHidden = 800
    # use 1 dropout to test the rbm for now
    net = rbm.RBM(nrVisible, nrHidden, 0.01, 1, 1,
                  binary=1-args.relu,
                  visibleActivationFunction=activationFunction,
                  hiddenActivationFunction=activationFunction,
                  rmsprop=args.rbmrmsprop,
                  nesterov=args.rbmnesterov)
    net.train(trainData)
    t = visualizeWeights(net.weights.T, SMALL_SIZE, (10,10))
  else:
    # Take the saved network and use that for reconstructions
    f = open(args.netFile, "rb")
    t = pickle.load(f)
    net = pickle.load(f)
    f.close()

  # get a random image and see it looks like
  # if reconstructRandom:
  #   test = np.random.random_sample(test.shape)

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
  print "done"

  if args.save:
    f = open(args.netFile, "wb")
    pickle.dump(t, f)
    pickle.dump(net, f)


"""
  Arguments:
    big: should the big or small images be used?
"""
def deepBeliefKanadeCV(big=False):
  data, labels = readKanade(big)

  data, labels = shuffle(data, labels)

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  if args.relu:
    activationFunction = relu
  else:
    activationFunction = T.nnet.sigmoid

  # TODO: try boosting for CV in order to increase the number of folds
  params =[(0.1, 0.1, 0.9), (0.1, 0.05, 0.9), (0.05, 0.01, 0.9), (0.05, 0.05, 0.9),
           (0.1, 0.1, 0.95), (0.1, 0.05, 0.95), (0.05, 0.01, 0.95), (0.05, 0.05, 0.95),
           (0.1, 0.1, 0.99), (0.1, 0.05, 0.99), (0.05, 0.01, 0.99), (0.05, 0.05, 0.99)]

  unsupervisedData = buildUnsupervisedDataSet()

  kf = cross_validation.KFold(n=len(data), k=len(params))
  bestCorrect = 0
  bestProbs = 0

  fold = 0
  for train, test in kf:

    trainData = data[train]
    trainLabels = labels[train]

    # TODO: this might require more thought
    net = db.DBN(5, [1200, 1500, 1500, 1500, 7],
               binary=1-args.relu,
               activationFunction=activationFunction,
               unsupervisedLearningRate=params[fold][0],
               supervisedLearningRate=params[fold][1],
               momentumMax=params[i][2],
               nesterovMomentum=args.nesterov,
               rbmNesterovMomentum=args.rbmnesterov,
               rmsprop=args.rmsprop,
               miniBatchSize=args.miniBatchSize,
               hiddenDropout=0.5,
               rbmHiddenDropout=0.5,
               visibleDropout=0.8,
               rbmVisibleDropout=1)

    net.train(trainData, trainLabels,
              maxEpochs=args.maxEpochs,
              validation=args.validation,
              unsupervisedData=unsupervisedData)

    probs, predicted = net.classify(data[test])

    actualLabels = labels[test]
    correct = 0
    errorCases = []

    for i in xrange(len(test)):
      print "predicted"
      print "probs"
      print probs[i]
      print predicted[i]
      print "actual"
      actual = actualLabels[i]
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


# TODO: shuffle training data for minibatches
def deepBeliefKanade(big=False):
  data, labels = readKanade(big, None)

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  # Random data for training and testing
  kf = cross_validation.KFold(n=len(data), n_folds=5)
  for train, test in kf:
    break

  if args.relu:
    activationFunction = relu
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95
  else:
    activationFunction = T.nnet.sigmoid
    unsupervisedLearningRate = 0.05
    supervisedLearningRate = 0.01
    momentumMax = 0.95

  trainData = data[train]
  trainLabels = labels[train]

  # TODO: this might require more thought
  net = db.DBN(5, [1200, 1500, 1500, 1500, 7],
             binary=1-args.relu,
             activationFunction=activationFunction,
             unsupervisedLearningRate=unsupervisedLearningRate,
             # is this not a bad learning rate?
             supervisedLearningRate=supervisedLearningRate,
             momentumMax=momentumMax,
             nesterovMomentum=args.nesterov,
             rbmNesterovMomentum=args.rbmnesterov,
             rmsprop=args.rmsprop,
             miniBatchSize=args.miniBatchSize,
             hiddenDropout=0.5,
             rbmHiddenDropout=0.5,
             visibleDropout=0.8,
             rbmVisibleDropout=1)

  unsupervisedData = buildUnsupervisedDataSet()


  net.train(trainData, trainLabels, maxEpochs=args.maxEpochs,
            validation=args.validation, unsupervisedData=unsupervisedData)

  probs, predicted = net.classify(data[test])

  actualLabels = labels[test]
  correct = 0
  errorCases = []

  for i in xrange(len(test)):
    print "predicted"
    print "probs"
    print probs[i]
    print predicted[i]
    print "actual"
    actual = actualLabels[i]
    print np.argmax(actual)
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(test)


def buildUnsupervisedDataSet():
  return np.vstack((
    # readCroppedYale(),
    readAttData(),
    readJaffe(),
    # readNottingham(),
    readAberdeen()))


def readKanade(big=False, folds=None, equalize=args.equalize):
  if not equalize:
    if big:
      files = glob.glob('kanade_150*.pickle')
    else:
      files = glob.glob('kanade_f*.pickle')

    if not folds:
      folds = range(1, 6)

    # Read the data from them. Sort out the files that do not have
    # the folds that we want
    # TODO: do this better (with regex in the file name)
    # DO not reply on the order returned
    files = [ files[x -1] for x in folds]

    data = np.array([])
    labels = np.array([])

    # TODO: do proper CV in which you use 4 folds for training and one for testing
    # at that time
    dataFolds = []
    labelFolds = []
    for filename in files:
      with open(filename, "rb") as  f:
        # Sort out the labels from the data
        # TODO: run the readKanade again tomorrow and change these idnices here
        dataAndLabels = pickle.load(f)
        foldData = dataAndLabels[:, 0:-1]
        print "foldData.shape"
        print foldData.shape
        foldLabels = dataAndLabels[:,-1]
        dataFolds.append(foldData)
        foldLabels = np.array(map(int, foldLabels))

        vectorLabels = labelsToVectors(foldLabels -1, 7)
        labelFolds.append(vectorLabels)

        print "foldLabels.shape"
        print vectorLabels.shape


    data = np.vstack(tuple(dataFolds))
    labels = np.vstack(tuple(labelFolds))
  else:
    if big:
      fileName = 'equalized_kanade_big.pickle'
    else:
      fileName = 'equalized_kanade_small.pickle'

    if not os.path.exists(fileName):
      equalizeKanade(big)

    with open(fileName, "wb") as  f:
      data = pickle.load(f)
      labels = pickle.load(f)

  return data, labels


def equalizeKanade(big=False):
  data, labels = readKanade(big=big, equalize=False)

  if big:
      fileName = 'equalized_kanade_big.pickle'
  else:
      fileName = 'equalized_kanade_small.pickle'

  data = np.array(map(lambda x: equalize(x), data))

  with open(fileName, "wb") as f:
    pickle.dump(data, f)
    pickle.dump(labels, f)

# TODO: get big, small as argument in order to be able to fit the resizing
def readCroppedYale(equalize=True):
  # PATH = "/data/mcr10/yaleb/CroppedYale"
  PATH = "/home/aela/uni/project/CroppedYale"

  imageFiles = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(PATH)
    for f in fnmatch.filter(files, '*.pgm')]

  # Filter out the ones that containt "amyes bient"
  imageFiles = [ x for x in imageFiles if not "Ambient" in x]

  images = []
  for f in imageFiles:
    img = cv2.imread(f, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    if equalize:
      img = equalize(img)

    img = resize(img, SMALL_SIZE)

    images += [img.reshape(-1)]

  return np.array(images)

def readAttData():
  PATH = "/data/mcr10/att"
  # PATH = "/home/aela/uni/project/code/pics/cambrdige_pics"

  imageFiles = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(PATH)
    for f in fnmatch.filter(files, '*.pgm')]

  images = []
  for f in imageFiles:
    img = cv2.imread(f, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    if args.equalize:
      img = equalize(img)
    img = resize(img, SMALL_SIZE)
    images += [img.reshape(-1)]


  return np.array(images)

def readCropEqualize(path, extension, doRecognition, equalize=True,
                     isColoured=False):
  dirforres = "detection-cropped"
  pathForCropped = os.path.join(path, dirforres)

  if doRecognition:
    if not os.path.exists(pathForCropped):
      os.makedirs(pathForCropped)

    imageFiles = [(os.path.join(dirpath, f), f)
      for dirpath, dirnames, files in os.walk(path)
      for f in fnmatch.filter(files, '*.' + extension)]

    images = []

    for fullPath, shortPath in imageFiles:
      # Do not do this for already cropped images
      if pathForCropped in fullPath:
        continue

      print fullPath
      img = cv2.imread(fullPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)

      # if isColoured:
      #   # TODO: do I always have to multiply by 255 in this case?
      #   # I think I need to do that for face detection
      #   img = color.rgb2gray(img)
      #   img = np.array(img * 255, dtype='uint8')

      if equalize:
        img = equalize(img)

      face = facedetection.cropFace(img)
      if not face == None:
        # Only do the resizing once you are done with the cropping of the faces
        face = resize(face, SMALL_SIZE)
        # Check that you are always saving them in the right format
        print "face.min"
        print face.min()

        print "face.max"
        print face.max()

        assert face.min() >=0 and face.max() <=1
        images += [face.reshape(-1)]

        # Save faces as files
        croppedFileName = os.path.join(pathForCropped, shortPath)
        io.imsave(croppedFileName, face)

  else:
    images = []
    imageFiles = [os.path.join(dirpath, f)
      for dirpath, dirnames, files in os.walk(pathForCropped)
      for f in fnmatch.filter(files, '*.' + extension)]

    for f in imageFiles:
      img = cv2.imread(f, cv2.CV_LOAD_IMAGE_GRAYSCALE)
      if type(img[0,0]) == np.uint8:
        print "rescaling unit"
      # if not (img.min() >=0.0 and img.max()<=1.000008):
        img = img / 255.0
      images += [img.reshape(-1)]

  print len(images)
  return np.array(images)

# This needs some thought: remove the cropped folder from path?

def readJaffe():
  PATH = "/data/mcr10/jaffe"
  # PATH = "/home/aela/uni/project/jaffe"
  return readCropEqualize(PATH , "tiff", args.facedetection, equalize=args.equalize,
                          isColoured=False)

def readNottingham():
  PATH = "/home/aela/uni/project/nottingham"
  # PATH = "/data/mcr10/nottingham"
  return readCropEqualize(PATH, "gif", args.facedetection, equalize=args.equalize,
                          isColoured=False)

def readAberdeen():
  PATH = "/data/mcr10/Aberdeen"
  # PATH = "/home/aela/uni/project/Aberdeen"
  return readCropEqualize(PATH, "jpg", args.facedetection, equalize=args.equalize,
                           isColoured=True)

def main():
  # deepBeliefKanade()

  # readNottingham()
  # readCroppedYale()
  # readJaffe()
  # readAttData()
  # readAberdeen()
  readKanade()
  # if args.rbm:
  #   rbmEmotions()
  # elif args.cv:
  #   deepBeliefKanadeCV()
  # elif args.db:
  #   deepBeliefKanade()


# You can also group the emotions into positive and negative to see
# if you can get better results (probably yes)
if __name__ == '__main__':
  import random
  print "FIXING RANDOMNESS"
  random.seed(6)
  np.random.seed(6)
  main()