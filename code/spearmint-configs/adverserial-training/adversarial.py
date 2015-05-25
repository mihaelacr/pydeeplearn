"""Spearmint for the DBN module in pydeeplearn."""

__author__ = "Mihaela Rosca"
__contact__ = "mihaela.c.rosca@gmail.com"

from lib import deepbelief as db
from lib.common import *
from lib.activationfunctions import *

from read import readmnist


PATH = "/data/mcr10/pydeeplearn/MNIST"
TRAIN = 10000
TEST = 1000

def trainDBN(unsupervisedLearningRate,
             supervisedLearningRate,
             visibleDropout,
             hiddenDropout,
             miniBatchSize,
             momentumMax,
             maxEpochs):
  print 'in trainDBN'
  trainVectors, trainLabels =\
    readmnist.read(0, TRAIN, digits=None, bTrain=True, path=PATH)

  testVectors, testLabels =\
      readmnist.read(TRAIN, TRAIN + TEST,
                     digits=None, bTrain=True, path=PATH)

  trainVectors, trainLabels = shuffle(trainVectors, trainLabels)
  print 'done reading'
  trainVectors = np.array(trainVectors, dtype='float')
  trainingScaledVectors = scale(trainVectors)

  testVectors = np.array(testVectors, dtype='float')
  testingScaledVectors = scale(testVectors)

  trainVectorLabels = labelsToVectors(trainLabels, 10)
  print 'done scaling data'
  print 'creating DBN'
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
                  visibleDropout=visibleDropout,
                  hiddenDropout=hiddenDropout,
                  weightDecayL1=0,
                  weightDecayL2=0,
                  rbmHiddenDropout=1.0,
                  rbmVisibleDropout=1.0,
                  miniBatchSize=miniBatchSize,
                  adversarial_training=True,
                  # TODO: make this a learned param
                  preTrainEpochs=100,
                  sparsityConstraintRbm=False,
                  sparsityTragetRbm=0.01,
                  sparsityRegularizationRbm=None)

  net.train(trainingScaledVectors, trainVectorLabels,
            maxEpochs=maxEpochs, validation=False)

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

