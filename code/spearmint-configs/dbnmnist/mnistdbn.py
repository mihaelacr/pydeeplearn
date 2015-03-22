"""Spearmint for the DBN module in pydeeplearn."""

__author__ = "Mihaela Rosca"
__contact__ = "mihaela.c.rosca@gmail.com"

import argparse

from lib import deepbelief as db
from lib.common import *

from read import readmnist

parser = argparse.ArgumentParser(description='digit recognition')
parser.add_argument('--path',dest='path', type = str, default="/data/mcr10/project/pydeeplearn/code/MNIST",
                    help="the path to the MNIST files")
arser.add_argument('--trainSize', type=int, default=100,
                    help='the number of tranining cases to be considered')
parser.add_argument('--testSize', type=int, default=10,
                    help='the number of testing cases to be considered')

args = parser.parse_args()

def trainDBN(unsupervisedLearningRate,
             supervisedLearningRate,
             visibleDropout,
             hiddenDropout,
             miniBatchSize,
             momentumMax,
             maxEpochs):
  trainVectors, trainLabels =\
    readmnist.read(0, args.trainSize, digits=None, bTrain=True, path=args.path)

  testVectors, testLabels =\
      readmnist.read(args.trainSize,  args.trainSize + args.testSize,
                     digits=None, bTrain=True, path=args.path)

  trainVectors, trainLabels = shuffle(trainVectors, trainLabels)

  trainVectors = np.array(trainVectors, dtype='float')
  trainingScaledVectors = scale(trainVectors)

  testVectors = np.array(testVectors, dtype='float')
  testingScaledVectors = scale(testVectors)

  trainVectorLabels = labelsToVectors(trainLabels, 10)

  net = db.DBN(5, [784, 1000, 1000, 1000, 10],
                  binary=False,
                  unsupervisedLearningRate=unsupervisedLearningRate,
                  supervisedLearningRate=supervisedLearningRate,
                  momentumMax=momentumMax,
                  nesterovMomentum=True,
                  rbmNesterovMomentum=True,
                  activationFunction=Rectified(),
                  rbmActivationFunctionVisible=Identity(),
                  rbmActivationFunctionHidden=RectifiedNoisy(),
                  rmsprop=True,
                  visibleDropout=0.8,
                  hiddenDropout=0.5,
                  weightDecayL1=0,
                  weightDecayL2=0,
                  rbmHiddenDropout=1.0,
                  rbmVisibleDropout=1.0,
                  miniBatchSize=miniBatchSize,
                  # TODO: make this a learned param
                  preTrainEpochs=100,
                  sparsityConstraintRbm=False,
                  sparsityTragetRbm=0.01,
                  sparsityRegularizationRbm=None)

  net.train(trainingScaledVectors, trainVectorLabels,
            maxEpochs=maxEpochs, validation=args.validation)

  proabilities, predicted = net.classify(testingScaledVectors)
  error = getClassificationError(predicted, testLabels)
  print "error", error
  return error


# Write a function like this called 'main'
def main(job_id, params):
  print 'params', params
  return trainDBN(unsupervisedLearningRate=params['unsupervisedLearningRate'][0],
                  supervisedLearningRate=params['supervisedLearningRate'][0],
                  visibleDropout=params['visibleDropout'][0],
                  hiddenDropout=params['hiddenDropout'][0],
                  miniBatchSize=params['miniBatchSize'][0],
                  momentumMax=params['momentumMax'][0],
                  maxEpochs=params['maxEpochs'][0])