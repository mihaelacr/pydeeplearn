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


def objective(args):
    activationFunction = Sigmoid()
    unsupervisedLearningRate = 0.01
    supervisedLearningRate = 0.05
    momentumMax = 0.95    
    nrLayers = 5
    layerSizes = 1000
    
    hiddenLayers = []
    for i in range(0,nrLayers-2):
        hiddenLayers.append(layerSizes)
    dbnLayers=hiddenLayers
    dbnLayers.insert(0,784)
    dbnLayers.append(10)        
    
    nrFold = 5
    training =  len(labels)
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
                     hiddenDropout=0.5,
                     visibleDropout=0.8,
                     rbmHiddenDropout=1.0,
                     rbmVisibleDropout=1.0,
                     weightDecayL1=0,
                     weightDecayL2=0,
                     preTrainEpochs=100)
        net.train(trainingData, vectorLabels,
                  maxEpochs=100, validation=True)  
                  


space = ( 
	hp.qloguniform( 'l1_dim', log( 10 ), log( 1000 ), 1 ), 
	hp.qloguniform( 'l2_dim', log( 10 ), log( 1000 ), 1 ),
	hp.loguniform( 'learning_rate', log( 1e-5 ), log( 1e-2 )),
	hp.uniform( 'momentum', 0.5, 0.99 ),
	hp.uniform( 'l1_dropout', 0.1, 0.9 ),
	hp.uniform( 'decay_factor', 1 + 1e-3, 1 + 1e-1 )
)


best = fmin( run_test, space, algo = tpe.suggest, max_evals = 50 )

print best
