import argparse
from sklearn import cross_validation

from similarity_utils import *
from readfacedatabases import *
import similarity

parser = argparse.ArgumentParser(description='digit recognition')
parser.add_argument('--relu', dest='relu',action='store_true', default=False,
                    help=("if true, trains the RBM or DBN with a rectified linear unit"))
parser.add_argument('--cv', dest='cv',action='store_true', default=False,
                    help=("if true, does cv"))
parser.add_argument('--testYaleMain', dest='testYaleMain',action='store_true', default=False,
                    help=("if true, tests the net with the Kanade databse"))
parser.add_argument('--diffsubjects', dest='diffsubjects',action='store_true', default=False,
                    help=("if true, trains a net with different test and train subjects"))
parser.add_argument('--emotionsdiff', dest='emotionsdiff',action='store_true', default=False,
                    help=("if true, trains a net to distinguish between emotions"))



args = parser.parse_args()


def similarityMain():
  trainData1, trainData2, testData1, testData2, similaritiesTrain, similaritiesTest = splitDataMultiPIESubject()

  print "training with dataset of size ", len(trainData1)
  print len(trainData1)

  print "testing with dataset of size ", len(testData1)

  print "training with ", similaritiesTrain.sum(), "positive examples"
  print "training with ", len(similaritiesTrain) - similaritiesTrain.sum(), "negative examples"


  print "testing with ", similaritiesTest.sum(), "positive examples"
  print "testing with ", len(similaritiesTest) - similaritiesTest.sum(), "negative examples"
  print len(testData1)

  simNet = similarity.SimilarityNet(learningRate=0.001,
                                    maxMomentum=0.95,
                                    binary=True,
                                    rbmNrVis=1200,
                                    rbmNrHid=1000,
                                    rbmLearningRate=0.005,
                                    rbmDropoutHid=1.0,
                                    rbmDropoutVis=1.0,
                                    trainingEpochsRBM=10)

  simNet.train(trainData1, trainData2, similaritiesTrain, epochs=200)

  res = simNet.test(testData1, testData2)

  predicted = res > 0.5

  correct = (similaritiesTest == predicted).sum() * 1.0 / len(res)

  print res

  print correct


def similarityMainTestYale():
  subjectsToImgs = readMultiPIESubjects()

  trainData1, trainData2, trainSubjects1, trainSubjects2 =\
    splitDataAccordingToSubjects(subjectsToImgs, None, imgsPerSubject=None)

  similaritiesTrain =  similarityDifferentSubjects(trainSubjects1, trainSubjects2)

  testData1, testData2, similaritiesTest = splitSimilarityYale()

  print "training with dataset of size ", len(trainData1)
  print len(trainData1)

  print "testing with dataset of size ", len(testData1)
  print len(testData1)

  print "training with ", similaritiesTrain.sum(), "positive examples"
  print "training with ", len(similaritiesTrain) - similaritiesTrain.sum(), "negative examples"

  print "testing with ", similaritiesTest.sum(), "positive examples"
  print "testing with ", len(similaritiesTest) - similaritiesTest.sum(), "negative examples"

  simNet = similarity.SimilarityNet(learningRate=0.001,
                                    maxMomentum=0.95,
                                    binary=True,
                                    rbmNrVis=1200,
                                    rbmNrHid=1000,
                                    rbmLearningRate=0.005,
                                    rbmDropoutHid=1.0,
                                    rbmDropoutVis=1.0,
                                    trainingEpochsRBM=10)

  simNet.train(trainData1, trainData2, similaritiesTrain)

  res = simNet.test(testData1, testData2)

  predicted = res > 0.5

  correct = (similaritiesTest == predicted).sum() * 1.0 / len(res)

  print res

  print correct

def similarityDifferentSubjectsMain():
  nrSubjects = 147
  subjects = np.array(range(nrSubjects))
  kf = cross_validation.KFold(n=len(subjects), n_folds=5)

  for train, test in kf:
    break


  subjectsToImgs = readMultiPIESubjects()

  subjectTrain = subjects[train]
  subjectTest = subjects[test]

  print "len(subjectTrain)"
  print len(subjectTrain)
  print "len(subjectTest)"
  print len(subjectTest)

  trainData1, trainData2, trainSubjects1, trainSubjects2 =\
    splitDataAccordingToSubjects(subjectsToImgs, subjectTrain, imgsPerSubject=None)


  testData1, testData2, testSubjects1, testSubjects2 =\
    splitDataAccordingToSubjects(subjectsToImgs, subjectTest, imgsPerSubject=None)

  print "training with dataset of size ", len(trainData1)
  print "testing with dataset of size ", len(testData1)

  similaritiesTrain =  similarityDifferentSubjects(trainSubjects1, trainSubjects2)
  similaritiesTest =  similarityDifferentSubjects(testSubjects1, testSubjects2)

  print "training with ", similaritiesTrain.sum(), "positive examples"
  print "training with ", len(similaritiesTrain) - similaritiesTrain.sum(), "negative examples"

  print "testing with ", similaritiesTest.sum(), "positive examples"
  print "testing with ", len(similaritiesTest) - similaritiesTest.sum(), "negative examples"

  simNet = similarity.SimilarityNet(learningRate=0.001,
                                    maxMomentum=0.95,
                                    binary=True,
                                    rbmNrVis=1200,
                                    rbmNrHid=1000,
                                    rbmLearningRate=0.005,
                                    rbmDropoutHid=1.0,
                                    rbmDropoutVis=1.0,
                                    trainingEpochsRBM=10)

  simNet.train(trainData1, trainData2, similaritiesTrain)

  res = simNet.test(testData1, testData2)

  predicted = res > 0.5

  correct = (similaritiesTest == predicted).sum() * 1.0 / len(res)

  print res

  print correct


def similarityCV():
  trainData1, trainData2, testData1, testData2, similaritiesTrain, similaritiesTest = splitDataMultiPIESubject()

  params = [(0.0001, 0.01), (0.0001, 0.005), (0.001, 0.01), (0.001, 0.005)]
  kf = cross_validation.KFold(n=len(trainData1), n_folds=len(params))

  fold = 0
  for train, test in kf:
    simNet = similarity.SimilarityNet(learningRate=params[fold][0],
                                    maxMomentum=0.95,
                                    binary=True,
                                    rbmNrVis=1200,
                                    rbmNrHid=1000,
                                    rbmLearningRate=params[fold][1],
                                    rbmDropoutHid=1.0,
                                    rbmDropoutVis=1.0)

    simNet.train(trainData1, trainData2, similaritiesTrain)

    print "training with ", similaritiesTrain.sum(), "positive examples"
    print "training with ", len(similaritiesTrain) - similaritiesTrain.sum(), "negative examples"

    print "testing with ", similaritiesTest.sum(), "positive examples"
    print "testing with ", len(similaritiesTest) - similaritiesTest.sum(), "negative examples"

    res = simNet.test(testData1, testData2)

    # error = (similaritiesTest - res)  * 1.0 / len(res)

    predicted = res > 0.5

    print "predicted"
    print predicted

    correct = (similaritiesTest == predicted).sum() * 1.0 / len(res)

    print "params[fold]"
    print params[fold]
    print "res"
    print res
    print "correct"
    print correct

    fold += 1


def similarityEmotionsMain():
  trainData1, trainData2, trainLabels, testData1, testData2, testLabels = splitSimilaritiesPIEEmotions()

  print "training with dataset of size ", len(trainData1)
  print len(trainData1)

  print "testing with dataset of size ", len(testData1)
  print len(testData1)

  # for k in testLabels:
  #   print k

  # for k in trainLabels:
  #   print k

  simNet = similarity.SimilarityNet(learningRate=0.001,
                                    maxMomentum=0.95,
                                    binary=True,
                                    rbmNrVis=1200,
                                    rbmNrHid=1000,
                                    rbmLearningRate=0.005,
                                    rbmDropoutHid=1.0,
                                    rbmDropoutVis=1.0,
                                    trainingEpochsRBM=10)

  print "training with ", trainLabels.sum(), "positive examples"
  print "training with ", len(trainLabels) - trainLabels.sum(), "negative examples"

  print "testing with ", testLabels.sum(), "positive examples"
  print "testing with ", len(testLabels) - testLabels.sum(), "negative examples"


  simNet.train(trainData1, trainData2, trainLabels, epochs=400)

  res = simNet.test(testData1, testData2)

  predicted = res > 0.5

  correct = (testLabels == predicted).sum() * 1.0 / len(res)

  print res

  print correct

def main():
  if args.cv:
    similarityCV()
  if args.diffsubjects:
    similarityDifferentSubjectsMain()
  if args.testYaleMain:
    similarityMainTestYale()
  if args.emotionsdiff:
    similarityEmotionsMain()
  else:
    similarityMain()

if __name__ == '__main__':
  main()
