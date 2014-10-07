# This module creates an optimization of hyper-parameters of a DBN using hyperopt library (https://github.com/hyperopt/hyperopt).
#  
# 

import numpy as np
import hyperopt

from hyperopt import hp, fmin, tpe
from sklearn import cross_validation
from lib import deepbelief as db
from read import readmnist
from lib.activationfunctions import *

# define an objective function
def objective(nrLayers,layerSizes,
              activationFunction,
              unsupervisedLearningRate,
              supervisedLearningRate,
              momentumMax,
              hiddenDropout,
              visibleDropout,
              rbmHiddenDropout,
              rbmVisibleDropout,
              weightDecayL1,
              weightDecayL2,
              preTrainEpochs,
              maxEpochs,
              trainingData,
              vectorLabels
              ):

    hiddenLayers = []
    for i in range(0,nrLayers-2):
        hiddenLayers.append(layerSizes)
    dbnLayers=hiddenLayers
    dbnLayers.insert(0,784)
    dbnLayers.append(10)        
    
    nrFolds = 5
    training =  len(vectorLabels)
    kf = cross_validation.kFold(n=training, n_folds=nrFolds)
    foldError =[]
    for training, testing in kf:
               
        net = db.DBN(nrLayers, dbnLayers,
                     binary=False,
                     unsupervisedLearningRate=unsupervisedLearningRate,
                     supervisedLearningRate=supervisedLearningRate,
                     momentumMax=momentumMax,
                     activationFunction=activationFunction,
                     rbmActivationFunctionVisible=activationFunction,
                     rbmActivationFunctionHidden=activationFunction,
                     nesterovMomentum=True,
                     rbmNesterovMomentum=True,
                     rmsprop=True,
                     hiddenDropout=hiddenDropout,
                     visibleDropout=visibleDropout,
                     rbmHiddenDropout=rbmHiddenDropout,
                     rbmVisibleDropout=rbmHiddenDropout,
                     weightDecayL1=weightDecayL1,
                     weightDecayL2=weightDecayL2,
                     preTrainEpochs= preTrainEpochs)
        net.train(trainingData, vectorLabels,
                  maxEpochs=maxEpochs, validation=False)


def main():
        activationFunction = Sigmoid()
    unsupervisedLearningRate = 0.01
    supervisedLearningRate = 0.05
    momentumMax = 0.95    
    nrLayers = 5
    layerSizes = 1000
    
    trainVectors, trainLabels =\
        readmnist.read(0, training, bTrain=True, path=args.path)
    testVectors, testLabels =\
        readmnist.read(0, testing, bTrain=False, path=args.path)
    print trainVectors[0].shape
    
    trainVectors, trainLabels = shuffle(trainVectors, trainLabels)
    
    activationFunction = Sigmoid()
    
    trainingScaledVectors = trainVectors / 255.0
    testingScaledVectors = testVectors / 255.0
    
    
    # define a search space
    space = ( 
	hp.qloguniform( 'l1_dim', log( 10 ), log( 1000 ), 1 ), 
	hp.qloguniform( 'l2_dim', log( 10 ), log( 1000 ), 1 ),
	hp.loguniform( 'learning_rate', log( 1e-5 ), log( 1e-2 )),
	hp.uniform( 'momentum', 0.5, 0.99 ),
	hp.uniform( 'l1_dropout', 0.1, 0.9 ),
	hp.uniform( 'decay_factor', 1 + 1e-3, 1 + 1e-1 )
    )
	
    # minimize the objective over the space
    best = fmin( run_test, space, algo = tpe.suggest, max_evals = 50 )
	
    print best
    
  
if __name__ == '__main__':
  main()

