import math

__author__ = 'snurkabill'

import csv
import cPickle as pickle
from lib import deepbelief as db
from lib.activationfunctions import *
from lib.trainingoptions import *

def read(path="___"):
    with open(path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='\n')
        inputVectors = []
        outputVectors = []
        sizeOfKnownPartOfVector = 8 * 12
        for row in reader:
            row = row[:(len(row) - 1)]
            inputVectors.append(np.array(row[:sizeOfKnownPartOfVector]))
            outputVectors.append(np.array(row[sizeOfKnownPartOfVector:]))
        return inputVectors, outputVectors

def restrictToyStockDataColumns(inputVectors, outputVectors, columnIndex):
    restrictedInputVectors = []
    restrictedOutputVectors = []

    datasetSize = len(inputVectors)
    inputVectorSize = len(inputVectors[0])
    outputVectorSize = len(outputVectors[0])

    i = 0
    while i < datasetSize:
        newInputVector = []
        newOutputVector = []
        j = 0
        while j < inputVectorSize:
            if (j - columnIndex) % 12 == 0:
                newInputVector.append(inputVectors[i][j])
            j += 1
        j = 0
        while j < outputVectorSize:
            if (j - columnIndex) % 12 == 0:
                newOutputVector.append(outputVectors[i][j])
            j += 1
        restrictedInputVectors.append(newInputVector)
        restrictedOutputVectors.append(newOutputVector)
        i += 1
    return restrictedInputVectors, restrictedOutputVectors

def calcMeans(vectors):
    means = []
    i = 0
    while i < len(vectors[0]):
        means.append(0.0)
        i += 1
    i = 0
    while i < len(vectors):
        j = 0
        while j < len(vectors[0]):
            means[j] += vectors[i][j]
            j += 1
        i += 1
    i = 0
    while i < len(means):
        means[i] /= len(vectors)
        i += 1
    return means

def calcDeviations(vectors, means):
    stdDev = []
    i = 0
    while i < len(vectors[0]):
        j = 0
        sum = 0.0
        while j < len(vectors):
            sum += (vectors[j][i] - means[i]) * (vectors[j][i] - means[i])
            j += 1
        stdDev.append(math.sqrt(sum/len(vectors)))
        i += 1
    return stdDev

def normalizeData(vectors, means, deviations):
    i = 0
    while i < len(vectors):
        j = 0
        while j < len(vectors[0]):
            vectors[i][j] = (vectors[i][j] - means[j]) / deviations[j]
            j += 1
        i += 1
    return vectors

def calcRootMeanSquare(targetVectors, outputVectors):
    rmse = []
    i = 0
    while i < len(targetVectors[0]):
        j = 0
        sum = 0
        while j < len(targetVectors):
            sum += (targetVectors[j][i] - outputVectors[j][i]) * (targetVectors[j][i] - outputVectors[j][i])
            j += 1
        sum /= len(targetVectors)
        rmse.append(math.sqrt(sum))
        i += 1
    return rmse

def calcGlobalCost(targetVectors, outputVectors):
    globalError = 0.0
    i = 0
    while i < len(targetVectors):
        instanceError = 0.0
        j = 0
        while j < len(targetVectors[0]):
            instanceError += (targetVectors[i][j] - outputVectors[i][j]) * (targetVectors[i][j] - outputVectors[i][j])
            j += 1
        globalError += instanceError
        i += 1
    return globalError * 0.5

def rightSign(targetVectors, outputVectors):
    sizeOfOutputVector = len(targetVectors[0])
    sizeOfTestingSet = len(targetVectors)
    rightSign = []
    i = 0
    while(i < sizeOfOutputVector):
        i += 1
        rightSign.append(0)
    i = 0
    while i < sizeOfTestingSet:
        j = 0
        instanceError = 0
        while j < sizeOfOutputVector:
            difference = targetVectors[i][j] - outputVectors[i][j]
            instanceError += difference * difference
            if targetVectors[i][j] < 0 and outputVectors[i][j] < 0:
                rightSign[j] += 1
            elif targetVectors[i][j] >= 0 and outputVectors[i][j] >= 0:
                rightSign[j] += 1
            j += 1
        i += 1

    i = 0
    while i < sizeOfOutputVector:
        rightSign[i] = (rightSign[i] * 100) / sizeOfTestingSet
        i += 1

    return rightSign

def evalReport(targetVectors, outputVectors):

    print "RMSE: " + str(calcRootMeanSquare(targetVectors, outputVectors))
    rights = rightSign(targetVectors, outputVectors)
    print "Right signs: " + str(rights)
    file = open('results_', 'w')
    for item in rights:
        file.write("%s " % item)
    globalError = calcGlobalCost(targetVectors, outputVectors)
    print "Total Global Error: " + str(globalError)
    print "Average Global Error: " + str(globalError/len(targetVectors))

def main():
    '''
    not sure how to use resources in python yet
    '''
    path = "/home/snurkabill/Git/FirstDataParseer/"
    inputVectors, outputVectors = read(path + "toystocks.tsv.parsed")

    '''inputVectors = np.array(inputVectors, dtype='float')
    outputVectors = np.array(outputVectors, dtype='float')'''

    assert len(inputVectors) == len(outputVectors)

    totalSize = len(inputVectors)

    trainingSize = (totalSize / 8) * 7
    testingSize = totalSize - trainingSize

    trainingInputVectors = inputVectors[:trainingSize]
    testingInputVectors = inputVectors[trainingSize:]

    trainingInputVectors = np.array(trainingInputVectors, dtype='float')
    testingInputVectors = np.array(testingInputVectors, dtype='float')

    trainingOutputVectors = outputVectors[:trainingSize]
    testingOutputVectors = outputVectors[trainingSize:]

    trainingOutputVectors = np.array(trainingOutputVectors, dtype='float')
    testingOutputVectors = np.array(testingOutputVectors, dtype='float')

    topology = [len(inputVectors[0]), 100, len(outputVectors[0])]
    net = db.DBN(len(topology), topology,
                 binary=False,
                 unsupervisedLearningRate=0.005,
                 supervisedLearningRate=0.0005,
                 momentumMax=0.97,
                 activationFunction=Rectified(),
                 rbmActivationFunctionVisible=Identity(),
                 rbmActivationFunctionHidden=RectifiedNoisy(),
                 nesterovMomentum=False,
                 rbmNesterovMomentum=False,
                 rmsprop=True,
                 hiddenDropout=1,
                 visibleDropout=1,
                 rbmHiddenDropout=1.0,
                 rbmVisibleDropout=1.0,
                 weightDecayL1=0.0,
                 weightDecayL2=0.0,
                 sparsityTragetRbm=0.01,
                 sparsityConstraintRbm=False,
                 sparsityRegularizationRbm=0.005,
                 preTrainEpochs=10,
                 classificationActivationFunction=Identity())

    net.train(trainingInputVectors, trainingOutputVectors,
              maxEpochs=5, validation=False)


    probs, predicted = net.classify(testingInputVectors)
    evalReport(testingOutputVectors, probs)
    print type(predicted)



if __name__ == '__main__':
    main()
