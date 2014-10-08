
# This module creates an optimization of hyper-parameters of a DBN using hyperopt library (https://github.com/hyperopt/hyperopt).
# command
# python hyperopt_exampleMNIST.py --trainSize=10000 --path=/home/hugo/Desktop/pydeeplearn-master/code/MNIST-data/
import theano
import argparse
import numpy as np
from  math import log
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn import cross_validation

from lib import deepbelief as db
from lib.common import *
from lib.activationfunctions import Sigmoid


from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from read import readmnist


theanoFloat = theano.config.floatX


parser = argparse.ArgumentParser(description='digit recognition')
parser.add_argument('--trainSize', type=int, default=10000,
help='the number of tranining cases to be considered')
parser.add_argument('--path',dest='path', default="MNIST", help="the path to the MNIST files")
args = parser.parse_args()

# define an objective function
def objective( x ):
    
    global run_counter
    run_counter += 1
    print "run {}".format( run_counter )
    print "{}\n".format( x )    
    
    nrLayers = x[0]
    layerSizes = x[1]
    unsupervisedLearningRate = x[2]
    supervisedLearningRate = x[3]
    momentumMax = x[4]
    visibleDropout = x[5]      
    
    activationFunction = Sigmoid()
    rbmHiddenDropout = 1.0
    rbmVisibleDropout = 1.0
    weightDecayL1 = 0
    weightDecayL2 = 0
    preTrainEpochs = 1
    maxEpochs = 10  #Change here
    hiddenDropout = 0.5
    hiddenLayers = []
    
    #Using constant width net as Larochelle Exploring Strategies for Training Deep Neural Networks
    for i in range(0,nrLayers-2):
        hiddenLayers.append(layerSizes)
        
    dbnLayers=hiddenLayers
    dbnLayers.insert(0,784)
    dbnLayers.append(10)
    nrFolds = 5
    training = len(train_label)
    kf = cross_validation.KFold(training, n_folds=nrFolds)
    UACROC =[]
    PRECISION =[]
    RECALL =[]
    FSCORE =[]
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
                     rbmVisibleDropout=rbmVisibleDropout,
                     weightDecayL1=weightDecayL1,
                     weightDecayL2=weightDecayL2,
                     preTrainEpochs= preTrainEpochs)
        net.train(train_data[training], train_label[training],
                  maxEpochs=maxEpochs, validation=False)
        
        proabilities, predicted = net.classify(train_data[testing])
        vectorPredicted = labelsToVectors(predicted, 10)
        testLabels = train_label[testing]
        UACROC.append(roc_auc_score(testLabels, proabilities))                
        [pre,rec,fsc,_] = precision_recall_fscore_support(testLabels,vectorPredicted, average='macro')
        PRECISION.append(pre)
        RECALL.append(rec)
        FSCORE.append(fsc)
        
    return {
            'loss': 1-np.mean(UACROC),
            'status': STATUS_OK,             
            'classifier_precision': {'type': float, 'value': np.mean(PRECISION)},
            'classifier_recall': {'type': float, 'value': np.mean(RECALL)},
            'classifier_fscore': {'type': float, 'value': np.mean(FSCORE)}
            }
    
def main():
   # from numpy import genfromtxt
    import random
    print "FIXING RANDOMNESS"
    random.seed(6)
    np.random.seed(6)

    
    global run_counter  
    run_counter = 0

    training = args.trainSize
    trainVectors, trainLabels =\
        readmnist.read(0, training, bTrain=True, path=args.path)
    print trainVectors[0].shape

    trainVectors, trainLabels = shuffle(trainVectors, trainLabels)       
    
    trainingScaledVectors = trainVectors / 255.0
    
    vectorLabels = labelsToVectors(trainLabels, 10)

    global train_data
    train_data = trainingScaledVectors
    global train_label
    train_label = vectorLabels
    # define a search space
    space = (
        hp.choice('nrLayers', [2,3]),
        hp.choice('layerSizes', [500,750,1000,1250]),
        hp.loguniform( 'unsupervisedLearningRate', log( 1e-5 ), log( 1e-1)),
        hp.loguniform( 'supervisedLearningRate', log( 1e-5 ), log( 1e-1 )),
        hp.uniform( 'momentumMax', 0.5, 0.99 ),
        hp.uniform( 'visibleDropout', 0.1, 0.9 )
    )
    # Or use MongoTrials    
    trials = Trials()
    # minimize the objective over the space
    # change max_eval for a better search
    best = fmin( objective,
                space,
                algo = tpe.suggest,
                max_evals = 2,
                trials=trials)
    print "Best Parameters\n"        
    print best
    print "\n"    
    for i in range(0,run_counter):
        print "{}\n".format(trials.results[i]) # a list of dictionaries returned by 'objective' during the search
    print "\n"
    print "{}".format(trials.losses())# a list of losses (float for each 'ok' trial)

    
if __name__ == '__main__':
    main()
