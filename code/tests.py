import cPickle as pickle
from sklearn import cross_validation
import argparse


import numpy as np

import deepbelief as db
import restrictedBoltzmannMachine as rbm

from common import *
from readfacedatabases import *
from activationfunctions import *

parser = argparse.ArgumentParser(description='tests')
parser.add_argument('netFile', help="file where the serialized network should be saved")
parser.add_argument('--relu', dest='relu',action='store_true', default=False,
                    help=("if true, trains the RBM or DBN with a rectified linear unit"))


# Get the arguments of the program
args = parser.parse_args()


def testPicklingDBN():
  data, labels = readKanade(False, None, equalize=False)

  print "data.shape"
  print data.shape
  print "labels.shape"
  print labels.shape

  # Random data for training and testing
  kf = cross_validation.KFold(n=len(data), n_folds=5)
  for train, test in kf:
    break

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

  trainData = data[train]
  trainLabels = labels[train]

  # TODO: this might require more thought
  net = db.DBN(5, [1200, 1500, 1500, 1500, 7],
             binary=1-args.relu,
             activationFunction=activationFunction,
             rbmActivationFunctionVisible=rbmActivationFunctionVisible,
             rbmActivationFunctionHidden=rbmActivationFunctionHidden,
             unsupervisedLearningRate=unsupervisedLearningRate,
             supervisedLearningRate=supervisedLearningRate,
             momentumMax=momentumMax,
             nesterovMomentum=True,
             rbmNesterovMomentum=True,
             rmsprop=True,
             miniBatchSize=20,
             hiddenDropout=0.5,
             visibleDropout=0.8,
             rbmVisibleDropout=1.0,
             rbmHiddenDropout=1.0,
             preTrainEpochs=1)

  net.train(trainData, trainLabels, maxEpochs=10,
            validation=False,
            unsupervisedData=None,
            trainingIndices=train)

  initialDict = net.__dict__


  with open(args.netFile, "wb") as f:
    pickle.dump(net, f)

  with open(args.netFile, "rb") as f:
    net = pickle.load(f)

  afterDict = net.__dict__

  del initialDict['rbmActivationFunctionHidden']
  del initialDict['rbmActivationFunctionVisible']

  del afterDict['rbmActivationFunctionHidden']
  del afterDict['rbmActivationFunctionVisible']


  for key in initialDict:
    assert key in afterDict
    if isinstance(initialDict[key], (np.ndarray, np.generic)):
      assert np.arrays_equal(initialDict[key], afterDict[key])
    else:
      assert initialDict[key] == afterDict[key]



def main():
  testPicklingDBN()

if __name__ == '__main__':
  main()


