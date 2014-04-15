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

import facedetection


import deepbelief as db
from common import *

parser = argparse.ArgumentParser(description='digit recognition')
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
parser.add_argument('--cv', dest='cv',action='store_true', default=False,
                    help=("if true, do cross validation"))

# DEBUG mode?
parser.add_argument('--debug', dest='debug',action='store_false', default=False,
                    help=("if true, the deep belief net is ran in DEBUG mode"))

# Get the arguments of the program
args = parser.parse_args()

# Set the debug mode in the deep belief net
db.DEBUG = args.debug

"""
  Arguments:
    big: should the big or small images be used?
    folds: which folds should be used (1,..5) (a list). If None is passed all
    folds are used
"""
def deepBeliefKanadeCV(big=False, folds=None):
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
  # TODO: do LDA on the training data

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


  data =  np.vstack(tuple(dataFolds))
  labels = np.vstack(tuple(labelFolds))

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  unsupervisedData = np.vstack((readCroppedYale(), readAttData(), readJaffe()))


  kf = cross_validation.KFold(n=len(data), k=len(folds))
  bestCorrect = 0
  bestProbs = 0

  # bestParam
  # 0.001
  # bestProbs
  # 0.3703703703
  # TODO: try boosting for CV in order to increase the number of folds
  params = [0.1, 0.01, 0.001, 0.0001, 0.00001]
  fold = 0
  for train, test in kf:

    trainData = data[train]
    trainLabels = labels[train]

    # TODO: this might require more thought
    net = db.DBN(5, [1200, 1000, 1000, 1000, 7],
               unsupervisedLearningRate=0.01,
               supervisedLearningRate=params[fold],
               nesterovMomentum=args.nesterov,
               rmsprop=args.rmsprop,
               hiddenDropout=0.5, rbmHiddenDropout=0.5, visibleDropout=0.8,
               rbmVisibleDropout=1)

    net.train(trainData, trainLabels, unsupervisedData=unsupervisedData)

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
      print actual
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


def deepBeliefKanade(big=False):
  if big:
    files = glob.glob('kanade_150*.pickle')
  else:
    files = glob.glob('kanade_f*.pickle')

  # Read the data from them. Sort out the files that do not have
  # the folds that we want
  # TODO: do this better (with regex in the file name)
  # DO not reply on the order returned

  data = np.array([])
  labels = np.array([])
  # TODO: do LDA on the training data

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


  data =  np.vstack(tuple(dataFolds))
  labels = np.vstack(tuple(labelFolds))

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape


  train = range(0, 350)
  test = range(350, len(data))

  trainData = data[train]
  trainLabels = labels[train]

  # TODO: this might require more thought
  net = db.DBN(5, [1200, 1000, 1000, 1000, 7],
             unsupervisedLearningRate=0.01,
             supervisedLearningRate=0.001,
             nesterovMomentum=args.nesterov,
             rmsprop=args.rmsprop,
             hiddenDropout=0.5, rbmHiddenDropout=0.5, visibleDropout=0.8,
             rbmVisibleDropout=1)

  # unsupervisedData = readCroppedYale()
  unsupervisedData = np.vstack((readCroppedYale(), readAttData(), readJaffe()))


  net.train(trainData, trainLabels, unsupervisedData=unsupervisedData)

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
    print actual
    if predicted[i] == np.argmax(actual):
      correct += 1
    else:
      errorCases.append(i)

  print "correct"
  print correct

  print "percentage correct"
  print correct  * 1.0/ len(test)


# TODO: get big, small as argument in order to be able to fit the resizing
def readCroppedYale():
  PATH = "/data/mcr10/yaleb/CroppedYale"
  # PATH = "/home/aela/uni/project/CroppedYale"

  imageFiles = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(PATH)
    for f in fnmatch.filter(files, '*.pgm')]

  # Filter out the ones that containt "amyes bient"
  imageFiles = [ x for x in imageFiles if not "Ambient" in x]

  images = []
  for f in imageFiles:
    img = io.imread(f)
    img = resize(img, (30, 40))
    images += [img.reshape(-1)]

  return np.array(images)

def readAttData():
  # PATH = "/data/mcr10/att"
  PATH = "/home/aela/uni/project/code/pics/cambrdige_pics"

  imageFiles = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(PATH)
    for f in fnmatch.filter(files, '*.pgm')]

  images = []
  for f in imageFiles:
    img = io.imread(f)
    img = resize(img, (30, 40))
    # images += [img.reshape(-1)]
    images += [img]

  print len(images)
  return np.array(images)

# TODO: best crop the images using openCV
def readJaffe():
  # PATH = "/data/mcr10/jaffe"
  PATH = "/home/aela/uni/project/jaffe"
  imageFiles = [os.path.join(dirpath, f)
    for dirpath, dirnames, files in os.walk(PATH)
    for f in fnmatch.filter(files, '*.tiff')]

  images = []
  for f in imageFiles:
    img = io.imread(f)
    print img.shape

    face = facedetection.cropFace(img)

    # Only do the resizing once you are done with the cropping of the faces
    img = resize(face, (30, 40))
    # images += [img.reshape(-1)]
    images += [img]

  print len(images)
  return np.array(images)


def main():
  if args.cv:
    deepBeliefKanadeCV()
  elif args.db:
    deepBeliefKanade()


# You can also group the emotions into positive and negative to see
# if you can get better results (probably yes)
if __name__ == '__main__':
  main()