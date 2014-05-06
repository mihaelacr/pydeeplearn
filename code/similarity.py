""" This module implements the idea of finding out emotions similarities
by using the experiments similar to what Hinton describes in his NRelu paper."""


import argparse
import restrictedBoltzmannMachine as rbm

import numpy as np
import theano
from theano import tensor as T
from sklearn import cross_validation

from common import *
from readfacedatabases import *

theanoFloat  = theano.config.floatX


parser = argparse.ArgumentParser(description='digit recognition')
parser.add_argument('--relu', dest='relu',action='store_true', default=False,
                    help=("if true, trains the RBM or DBN with a rectified linear unit"))

args = parser.parse_args()

class Trainer(object):

  def __init__(self, input1, input2, net):

    self.w = theano.shared(value=np.float32(0))
    self.b = theano.shared(value=np.float32(0))
    self.net = net

    theanoRng = RandomStreams(seed=np.random.randint(1, 1000))


    # Do I need to add all biases? Probably only the hidden ones
    # self.params = [self.w, self.b, self.net.sharedWeights] + self.net.sharedBiases
    self.params = [self.w, self.b]

    # This also has to be some theano graph
    hiddens1 = self.net.buildReconstructionForSymbolicVariable(input1, theanoRng)
    hiddens2 = self.net.buildReconstructionForSymbolicVariable(input2, theanoRng)

    cos = cosineDistance(hiddens1, hiddens2)

    prob = 1.0 /( 1.0 + T.exp(self.w * cos + self.b))

    self.output = prob


# TODO; you need to make this a class
# and you need to polish a lot of things
def similarity(data1, data2, similarities):
  miniBatchSize = 10
  epochs = 10

  miniBatchIndex = T.lscalar()

  data = np.vstack([data1, data2])
  activationFunction = T.nnet.sigmoid

  net = rbm.RBM(1200, 500, 0.001, 1, 1,
                  binary=1-args.relu,
                  visibleActivationFunction=activationFunction,
                  hiddenActivationFunction=activationFunction,
                  rmsprop=True,
                  nesterov=True)
  net.train(data)

  # The mini-batch data is a matrix
  x = T.matrix('x', dtype=theanoFloat)
  y = T.matrix('y', dtype=theanoFloat)
  z = T.vector('z', dtype=theanoFloat)

  trainer = Trainer(x, y, net)

  # this will be a given
  # similarities = getSimilarities(data1, data2)
  error = T.sum(T.sqr(trainer.output-z))

  learningRate = 0.01
  updates = []
  gradients = T.grad(error, trainer.params)
  for param, gradient in zip(trainer.params, gradients):
    newParam = param - learningRate * gradient
    updates.append((param, newParam))

  # Now you have to define the theano function
  discriminativeTraining = theano.function(
    inputs=[miniBatchIndex],
    output=[],
    givens={
          x: data1[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize],
          y: data2[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize],
          z: similarities[miniBatchIndex * miniBatchSize:(miniBatchIndex + 1) * miniBatchSize],
          })

  nrMiniBatches = len(data1) / miniBatchSize

  for epoch in xrange(epochs):
    for miniBatch in xrange(nrMiniBatches):
      discriminativeTraining(miniBatch)

  # now you also have to test it somehow
  # so you need to keep some testing data out


def cosineDistance(first, second):
  normFirst = T.sum(T.sqrt(first), axis=1)
  normSecond = T.sum(T.sqrt(second), axis=1)
  return T.sum(first * second, axis=1) / (normFirst * normSecond)


# you can create more tuples than just one per image
# you can put each image in 5 tuples and that will probably owrk better
# it might be useful to also give the same image twice
def splitData():
  subjectsToImgs = readMultiPIESubjects()

  data1 = []

  data2 = []

  shuffling = []
  subjectsShuffling = []
  subjects1 = []
  subjects2 = []

  for subject, images in subjectsToImgs.iteritems():
    lastIndex = len(images)/ 4 + subject % 2
    delta = len(images)/ 4 + (subject + 1) % 2
    last2Index = lastIndex + delta
    data1 += images[0: lastIndex]
    data2 += images[lastIndex: last2Index]

    subjects1 += [subject] * lastIndex
    subjects2 += [subject] * delta

    imagesForShuffling = images[last2Index : ]
    shuffling += imagesForShuffling
    subjectsShuffling += [subject] * len(imagesForShuffling)

  print "len(subjectsShuffling)"
  print len(subjectsShuffling)

  print "shuffling"
  print len(shuffling)

  assert len(shuffling) == len(subjectsShuffling)
  shuffling, subjectsShuffling = shuffleList(shuffling, subjectsShuffling)

  print len(data1)
  print len(data2)
  # Warning: hack
  data2 = data2[:-1]
  subjects2 = subjects2[:-1]

  data1 = np.array(data1)
  data2 = np.array(data2)
  subjects1 = np.array(subjects1)
  subjects2 = np.array(subjects2)

  trainData1, testData1, trainData2, testData2, trainSubjects1, testSubjects1,\
        trainSubjects2, testSubjects2 = splitTrainTest(data1, data2, subjects1, subjects2, 5)

  shuffledData1 = shuffling[0: len(shuffling) / 2]
  shuffledData2 = shuffling[len(shuffling)/2 :]

  subjectsData1 = subjectsShuffling[0: len(shuffling) /2]
  subjectsData2 = subjectsShuffling[len(shuffling)/2:]

  shuffledData2 = shuffledData2[:-1]
  subjectsData2 = subjectsData2[:-1]

  shuffledData1 = np.array(shuffledData1)
  shuffledData2 = np.array(shuffledData2)
  subjectsData1 = np.array(subjectsData1)
  subjectsData2 = np.array(subjectsData2)

  print len(shuffledData1)
  print len(shuffledData2)

  trainShuffedData1, testShuffedData1, trainShuffedData2, testShuffedData2,\
    trainShuffledSubjects1, testShuffledSubjects1, trainShuffledSubjects2, testShuffledSubjects2 =\
        splitTrainTest(shuffledData1, shuffledData2,
                      subjectsData1, subjectsData2, 5)

  trainData1 = np.vstack((trainData1, trainShuffedData1))
  trainData2 = np.vstack((trainData2, trainShuffedData2))

  testData1 = np.vstack((testData1, testShuffedData1))
  testData2 = np.vstack((testData2, testShuffedData2))

  trainSubjects1 = np.vstack((trainSubjects1, trainShuffledSubjects1))
  testSubjects1 = np.vstack((testSubjects1, testShuffledSubjects1))

  trainSubjects2 = np.vstack((trainSubjects2, trainShuffledSubjects2))
  testSubjects2 = np.vstack((testSubjects2, testShuffledSubjects2))

  assert len(subjects1) == len(subjects2)
  assert len(trainSubjects1) == len(trainSubjects1)
  assert len(testSubjects1) == len(testSubjects2)
  similaritiesTrain = (trainSubjects1 == trainSubjects1)
  similaritiesTest = (testSubjects1 == testSubjects2)

  assert len(trainData1) == len(trainData2)
  assert len(testData1) == len(testData2)

  return trainData1, trainData2, testData1, testData2, similaritiesTrain, similaritiesTest

def defineSimilartyMesures():
  None

# TODO: move to common?
def splitTrainTest(data1, data2, labels1, labels2, ratio):
  assert len(data1) == len(data2)
  assert len(labels1) == len(labels2)
  assert len(labels1) == len(data1)
  # Random data for training and testing
  kf = cross_validation.KFold(n=len(data1), k=ratio)
  for train, test in kf:
    break

  return (data1[train], data1[test], data2[train], data2[test],
          labels1[train], labels1[test], labels2[train], labels2[test])

def main():
  trainData1, trainData2, testData1, testData2, similaritiesTrain, similaritiesTest = splitData()
  similarity(trainData1, trainData2, similaritiesTrain)

if __name__ == '__main__':
  main()
