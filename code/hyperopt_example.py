# This module creates an optimization of hyper-parameters of a DBN using hyperopt library (https://github.com/hyperopt/hyperopt).
#  
# 
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

from hyperopt import hp, fmin, tpe

from sklearn import cross_validation

from lib import deepbelief as db
from lib import restrictedBoltzmannMachine as rbm
from lib.common import *
from lib.activationfunctions import *
from lib.trainingoptions import *

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support

theanoFloat  = theano.config.floatX

train_data = []
train_label = []

parser = argparse.ArgumentParser(description='digit recognition')
parser.add_argument('--trainSize', type=int, default=10000,
                    help='the number of tranining cases to be considered')
parser.add_argument('--path',dest='path', default="MNIST", help="the path to the MNIST files")
args = parser.parse_args()

# define an objective function
def objective(nrLayers,layerSizes,
              unsupervisedLearningRate,
              supervisedLearningRate,
              momentumMax,
              visibleDropout,
              ):
            
            
    activationFunction = Sigmoid()  
    rbmHiddenDropout = 1.0
    rbmVisibleDropout = 1.0    
    weightDecayL1 = 0
    weightDecayL2 = 0 
    preTrainEpochs = 1
    maxEpochs = 100
    hiddenDropout = 0.5
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
        net.train(train_data[training], train_label[training],
                  maxEpochs=maxEpochs, validation=False)
                  
        proabilities, predicted = net.classify(train_data[testing])
        testLabels = train_label[testing]
        
        UACROC.append(roc_auc_score(testLabels, proabilities))            
    
    return 1-np.mean(UACROC)


def main():
    from numpy import genfromtxt    
    import random
    print "FIXING RANDOMNESS"
    random.seed(6)
    np.random.seed(6)    
    
    training = args.trainSizeactivationFunction

    trainVectors, trainLabels =\
        readmnist.read(0, training, bTrain=True, path=args.path)

    trainVectors, trainLabels = shuffle(trainVectors, trainLabels)    
    trainingScaledVectors = trainVectors / 255.0          
    global train_data = trainingScaledVectors
    global train_label = trainLabels   
  
 # define a search space
    space = ( 
      hp.choice('nrLayers', [2,3])
      hp.choice('layerSizes', [500,600,700,800,900,1000,1100,1200])
      hp.loguniform( 'unsupervisedLearningRate', log( 1e-5 ), log( 1e-1)),
      hp.loguniform( 'supervisedLearningRate', log( 1e-5 ), log( 1e-1 )),
      hp.uniform( 'momentumMax', 0.5, 0.99 ),
      hp.uniform( 'visibleDropout',  0.1, 0.9 ),
    )
# minimize the objective over the space
    best = fmin( objective, space, algo = tpe.suggest, max_evals = 50 )
	
    print best
  
if __name__ == '__main__':
  main()
