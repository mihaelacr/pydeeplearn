import argparse
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

from similarity_utils import *
from activationfunctions import *
from readfacedatabases import *
import similarity

parser = argparse.ArgumentParser(description='digit recognition')
parser.add_argument('--relu', dest='relu',action='store_true', default=False,
                    help=("if true, trains the RBM with a rectified linear units"))
parser.add_argument('--sparsity', dest='sparsity',action='store_true', default=False,
                    help=("if true, trains the RBM with a sparsity constraint"))
parser.add_argument('--rmsprop', dest='rmsprop',action='store_true', default=False,
                    help=("if true, trains the similarity net is "))
parser.add_argument('--cv', dest='cv',action='store_true', default=False,
                    help=("if true, does cv"))
parser.add_argument('--cvEmotion', dest='cvEmotion',action='store_true', default=False,
                    help=("if true, does cv for emotions"))
parser.add_argument('--testYaleMain', dest='testYaleMain',action='store_true', default=False,
                    help=("if true, tests the net with the Kanade databse"))
parser.add_argument('--diffsubjects', dest='diffsubjects',action='store_true', default=False,
                    help=("if true, trains a net with different test and train subjects"))
parser.add_argument('--emotionsdiff', dest='emotionsdiff',action='store_true', default=False,
                    help=("if true, trains a net to distinguish between emotions"))
parser.add_argument('--emotionsdiffsamesubj', dest='emotionsdiffsamesubj',action='store_true', default=False,
                    help=("if true, trains a net to distinguish between emotions where the pictures presented are the same people"))
parser.add_argument('--equalize',dest='equalize',action='store_true', default=False,
                    help="if true, the input images are equalized before being fed into the net")
parser.add_argument('--nrHidden',dest='nrHidden', type=int, default=1000,
                    help="how many hidden units should be used for the net")
parser.add_argument('--epochs', type=int, default=500,
                    help='the maximum number of supervised epochs')
parser.add_argument('--rbmepochs', type=int, default=10,
                    help='the maximum number of unsupervised epochs')


args = parser.parse_args()


def similarityMain():
  trainData1, trainData2, testData1, testData2, similaritiesTrain, similaritiesTest =\
     splitDataMultiPIESubject(instanceToPairRatio=2, equalize=args.equalize)

  print "training with dataset of size ", len(trainData1)
  print len(trainData1)

  print "testing with dataset of size ", len(testData1)

  print "training with ", similaritiesTrain.sum(), "positive examples"
  print "training with ", len(similaritiesTrain) - similaritiesTrain.sum(), "negative examples"


  print "testing with ", similaritiesTest.sum(), "positive examples"
  print "testing with ", len(similaritiesTest) - similaritiesTest.sum(), "negative examples"
  print len(testData1)

  if args.relu:
    if args.rmsprop:
      learningRate = 0.005
      rbmLearningRate = 0.005
      maxMomentum = 0.95
    else:
      learningRate = 0.005
      rbmLearningRate = 0.005
      maxMomentum = 0.95

    visibleActivationFunction = Identity()
    hiddenActivationFunction = RectifiedNoisy()
    # IMPORTANT: SCALE THE DATA IF YOU USE GAUSSIAN VISIBlE UNITS
    testData1 = scale(testData1)
    testData2 = scale(testData2)
    trainData1 = scale(trainData1)
    trainData2 = scale(trainData2)

  # Stochastic binary units
  else:
    if args.rmsprop:
      learningRate = 0.001
      rbmLearningRate = 0.005
      maxMomentum = 0.95
    else:
      learningRate = 0.001
      rbmLearningRate = 0.05
      maxMomentum = 0.95

    visibleActivationFunction = Sigmoid()
    hiddenActivationFunction = Sigmoid()

  simNet = similarity.SimilarityNet(learningRate=learningRate,
                                    maxMomentum=maxMomentum,
                                    visibleActivationFunction=visibleActivationFunction,
                                    hiddenActivationFunction=hiddenActivationFunction,
                                    rbmNrVis=1200,
                                    rbmNrHid=args.nrHidden,
                                    rbmLearningRate=rbmLearningRate,
                                    rbmDropoutHid=1.0,
                                    rbmDropoutVis=0.8,
                                    rmsprop=False,
                                    trainingEpochsRBM=args.rbmepochs,
                                    nesterovRbm=True,
                                    sparsityConstraint=args.sparsity,
                                    sparsityRegularization=0.01,
                                    sparsityTraget=0.01)

  simNet.train(trainData1, trainData2, similaritiesTrain, epochs=args.epochs)

  res = simNet.test(testData1, testData2)

  # Try to change this threshold?
  predicted = res > 0.5

  correct = (similaritiesTest == predicted).sum() * 1.0 / len(res)

  confMatrix = confusion_matrix(similaritiesTest, predicted)
  print confMatrix

  print "correct"
  print correct


def similarityMainTestYale():
  subjectsToImgs = readMultiPIESubjects(args.equalize)

  trainData1, trainData2, trainSubjects1, trainSubjects2 =\
    splitDataAccordingToLabels(subjectsToImgs, None, instanceToPairRatio=2)

  similaritiesTrain =  similarityDifferentLabels(trainSubjects1, trainSubjects2)

  testData1, testData2, similaritiesTest = splitSimilarityYale(1, args.equalize)

  print "training with dataset of size ", len(trainData1)
  print len(trainData1)

  print "testing with dataset of size ", len(testData1)
  print len(testData1)

  print "training with ", similaritiesTrain.sum(), "positive examples"
  print "training with ", len(similaritiesTrain) - similaritiesTrain.sum(), "negative examples"

  print "testing with ", similaritiesTest.sum(), "positive examples"
  print "testing with ", len(similaritiesTest) - similaritiesTest.sum(), "negative examples"

  if args.relu:
    if args.rmsprop:
      learningRate = 0.005
      rbmLearningRate = 0.005
      maxMomentum = 0.95
    else:
      learningRate = 0.005
      rbmLearningRate = 0.005
      maxMomentum = 0.95

    visibleActivationFunction = Identity()
    hiddenActivationFunction = RectifiedNoisy()
    # IMPORTANT: SCALE THE DATA IF YOU USE GAUSSIAN VISIBlE UNITS
    testData1 = scale(testData1)
    testData2 = scale(testData2)
    trainData1 = scale(trainData1)
    trainData2 = scale(trainData2)

  else:
    if args.rmsprop:
      learningRate = 0.001
      rbmLearningRate = 0.005
      maxMomentum = 0.95
    else:
      learningRate = 0.001
      rbmLearningRate = 0.005
      maxMomentum = 0.95

    visibleActivationFunction = Sigmoid()
    hiddenActivationFunction = Sigmoid()


  simNet = similarity.SimilarityNet(learningRate=learningRate,
                                    maxMomentum=maxMomentum,
                                    visibleActivationFunction=visibleActivationFunction,
                                    hiddenActivationFunction=hiddenActivationFunction,
                                    rbmNrVis=1200,
                                    rbmNrHid=args.nrHidden,
                                    rbmLearningRate=rbmLearningRate,
                                    rbmDropoutHid=1.0,
                                    rbmDropoutVis=1.0,
                                    rmsprop=False,
                                    trainingEpochsRBM=args.rbmepochs,
                                    nesterovRbm=True,
                                    sparsityConstraint=args.sparsity,
                                    sparsityRegularization=0.01,
                                    sparsityTraget=0.01)

  simNet.train(trainData1, trainData2, similaritiesTrain, epochs=args.epochs)

  res = simNet.test(testData1, testData2)

  predicted = res > 0.5

  correct = (similaritiesTest == predicted).sum() * 1.0 / len(res)

  confMatrix = confusion_matrix(similaritiesTest, predicted)
  print confMatrix
  print correct

def similarityDifferentSubjectsMain():
  nrSubjects = 147
  subjects = np.array(range(nrSubjects))
  kf = cross_validation.KFold(n=len(subjects), n_folds=5)

  for train, test in kf:
    break

  subjectsToImgs = readMultiPIESubjects(args.equalize)

  subjectTrain = subjects[train]
  subjectTest = subjects[test]

  print "len(subjectTrain)"
  print len(subjectTrain)
  print "len(subjectTest)"
  print len(subjectTest)

  trainData1, trainData2, trainSubjects1, trainSubjects2 =\
    splitDataAccordingToLabels(subjectsToImgs, subjectTrain, instanceToPairRatio=2)


  testData1, testData2, testSubjects1, testSubjects2 =\
    splitDataAccordingToLabels(subjectsToImgs, subjectTest, instanceToPairRatio=2)

  print "training with dataset of size ", len(trainData1)
  print "testing with dataset of size ", len(testData1)

  similaritiesTrain =  similarityDifferentLabels(trainSubjects1, trainSubjects2)
  similaritiesTest =  similarityDifferentLabels(testSubjects1, testSubjects2)

  print "training with ", similaritiesTrain.sum(), "positive examples"
  print "training with ", len(similaritiesTrain) - similaritiesTrain.sum(), "negative examples"

  print "testing with ", similaritiesTest.sum(), "positive examples"
  print "testing with ", len(similaritiesTest) - similaritiesTest.sum(), "negative examples"

  if args.relu:
    if args.rmsprop:
      learningRate = 0.005
      rbmLearningRate = 0.005
      maxMomentum = 0.95
    else:
      learningRate = 0.005
      rbmLearningRate = 0.005
      maxMomentum = 0.95

    visibleActivationFunction = Identity()
    hiddenActivationFunction = RectifiedNoisy()
    # IMPORTANT: SCALE THE DATA IF YOU USE GAUSSIAN VISIBlE UNITS
    testData1 = scale(testData1)
    testData2 = scale(testData2)
    trainData1 = scale(trainData1)
    trainData2 = scale(trainData2)

  else:
    if args.rmsprop:
      learningRate = 0.001
      rbmLearningRate = 0.005
      maxMomentum = 0.95
    else:
      learningRate = 0.001
      rbmLearningRate = 0.005
      maxMomentum = 0.95

    visibleActivationFunction = Sigmoid()
    hiddenActivationFunction = Sigmoid()

  simNet = similarity.SimilarityNet(learningRate=learningRate,
                                    maxMomentum=maxMomentum,
                                    visibleActivationFunction=visibleActivationFunction,
                                    hiddenActivationFunction=hiddenActivationFunction,
                                    rbmNrVis=1200,
                                    rbmNrHid=args.nrHidden,
                                    rbmLearningRate=rbmLearningRate,
                                    rbmDropoutHid=1.0,
                                    rmsprop=False,
                                    rbmDropoutVis=1.0,
                                    trainingEpochsRBM=args.rbmepochs,
                                    nesterovRbm=True,
                                    sparsityConstraint=args.sparsity,
                                    sparsityRegularization=0.001,
                                    sparsityTraget=0.01)

  simNet.train(trainData1, trainData2, similaritiesTrain, epochs=args.epochs)

  res = simNet.test(testData1, testData2)

  predicted = res > 0.5

  correct = (similaritiesTest == predicted).sum() * 1.0 / len(res)

  confMatrix = confusion_matrix(similaritiesTest, predicted)
  print confMatrix


  print correct

def similarityCV():
  trainData1, trainData2, testData1, testData2, similaritiesTrain, similaritiesTest =\
     splitDataMultiPIESubject(instanceToPairRatio=2, equalize=args.equalize)

  if args.relu:
    visibleActivationFunction = Identity()
    hiddenActivationFunction = RectifiedNoisy()
    # IMPORTANT: SCALE THE DATA IF YOU USE GAUSSIAN VISIBlE UNITS
    testData1 = scale(testData1)
    testData2 = scale(testData2)
    trainData1 = scale(trainData1)
    trainData2 = scale(trainData2)
  else:
    visibleActivationFunction = Sigmoid()
    hiddenActivationFunction = Sigmoid()

  if args.relu:
    if args.rmsprop:
      params = [(0.001, 0.005), (0.001, 0.001), (0.005, 0.001), (0.005, 0.005)]
                # (0.001, 0.005), (0.001, 0.001), (0.005, 0.001), (0.005, 0.005),
                # (0.001, 0.005), (0.001, 0.001), (0.005, 0.001), (0.005, 0.005)]
      # params = [(0.001, 0.005, 0.1),    (0.001, 0.001, 0.1), (0.005, 0.001, 0.001), (0.005, 0.005, 0.1),
      #           (0.001, 0.005, 0.01),   (0.001, 0.001, 0.01), (0.005, 0.001, 0.001), (0.005, 0.005, 0.01),
      #           (0.001, 0.005, 0.001),  (0.001, 0.001, 0.001), (0.005, 0.001, 0.001), (0.005, 0.005, 0.001),
      #           (0.001, 0.005, 0.0001), (0.001, 0.001, 0.0001), (0.005, 0.001, 0.001), (0.005, 0.005, 0.0001)]
    else:
      params = [(0.001, 0.005, 0.1), (0.001, 0.001, 0.01), (0.005, 0.001, 0.001), (0.005, 0.005, 0.0001),
                (0.001, 0.005, 0.1), (0.001, 0.001, 0.01), (0.005, 0.001, 0.001), (0.005, 0.005, 0.0001),
                (0.001, 0.005, 0.1), (0.001, 0.001, 0.01), (0.005, 0.001, 0.001), (0.005, 0.005, 0.0001)]

  else:
    if args.rmsprop:
      params = [(0.0001, 0.01, 0.01), (0.0001, 0.005, 0.01), (0.001, 0.01, 0.01), (0.001, 0.005, 0.01)]
    else:
      params = [(0.01, 0.01), (0.01, 0.005), (0.0001, 0.05), (0.001, 0.005)]

  kf = cross_validation.KFold(n=len(trainData1), n_folds=len(params))

  correctForParams = []

  # Try bigger values for the number of units: 2000?
  fold = 0
  for train, test in kf:
    simNet = similarity.SimilarityNet(learningRate=params[fold][0],
                                      maxMomentum=0.95,
                                      visibleActivationFunction=visibleActivationFunction,
                                      hiddenActivationFunction=hiddenActivationFunction,
                                      rbmNrVis=1200,
                                      rbmNrHid=args.nrHidden,
                                      rbmLearningRate=params[fold][1],
                                      rbmDropoutHid=0.5,
                                      rmsprop=False,
                                      rbmDropoutVis=0.8,
                                      trainingEpochsRBM=args.rbmepochs,
                                      nesterovRbm=True,
                                      sparsityConstraint=args.sparsity,
                                      sparsityRegularization=params[fold][-1],
                                      sparsityTraget=0.01)

    simNet.train(trainData1, trainData2, similaritiesTrain, epochs=args.epochs)

    res = simNet.test(testData1, testData2)

    predicted = res > 0.5

    print "predicted"
    print predicted

    correct = (similaritiesTest == predicted).sum() * 1.0 / len(res)

    print "params[fold]"
    print params[fold]

    print "correct"
    print correct
    correctForParams += [correct]

    fold += 1

  for i in xrange(len(params)):
    print "parameter tuple " + str(params[i]) + " achieved correctness of " + str(correctForParams[i])


def similarityCVEmotions():
  data1, data2, labels = splitSimilaritiesPIE(instanceToPairRatio=2, equalize=args.equalize)

  if args.relu:
    visibleActivationFunction = Identity()
    hiddenActivationFunction = RectifiedNoisy()
    # IMPORTANT: SCALE THE DATA IF YOU USE GAUSSIAN VISIBlE UNITS
    # I am now doing it in the rbm level so it is not as important anymore to do it here
    data1 = scale(data1)
    data2 = scale(data2)
  else:
    visibleActivationFunction = Sigmoid()
    hiddenActivationFunction = Sigmoid()

  if args.relu:
    if args.rmsprop:
      # params = [(0.001, 0.01), (0.001, 0.005), (0.001, 0.1), (0.001, 0.05)]
      params = [ (0.005, 0.01), (0.005, 0.005), (0.005, 0.05)]
    else:
      params = [(0.001, 0.01), (0.001, 0.005), (0.001, 0.1), (0.001, 0.05)]

  else:
    if args.rmsprop:
      params = [(0.0001, 0.01), (0.0001, 0.005), (0.001, 0.01), (0.001, 0.005)]
    else:
      params = [(0.0001, 0.01), (0.0001, 0.005), (0.001, 0.01), (0.001, 0.005)]

  kf = cross_validation.KFold(n=len(data1), n_folds=len(params))

  correctForParams = []

  # Try bigger values for the number of units: 2000?
  fold = 0
  for train, test in kf:
    trainData1 = data1[train]
    trainData2 = data2[train]
    trainLabels = labels[train]

    testData1 = data1[test]
    testData2 = data2[test]
    testLabels = labels[test]


    simNet = similarity.SimilarityNet(learningRate=params[fold][0],
                                      maxMomentum=0.95,
                                      visibleActivationFunction=visibleActivationFunction,
                                      hiddenActivationFunction=hiddenActivationFunction,
                                      rbmNrVis=1200,
                                      rbmNrHid=args.nrHidden,
                                      rbmLearningRate=params[fold][1],
                                      rbmDropoutHid=1.0,
                                      rbmDropoutVis=1.0,
                                      rmsprop=False,
                                      trainingEpochsRBM=args.rbmepochs,
                                      nesterovRbm=True,
                                      sparsityConstraint=args.sparsity,
                                      sparsityRegularization=params[fold][-1],
                                      sparsityTraget=0.01)

    simNet.train(trainData1, trainData2, trainLabels, epochs=args.epochs)

    res = simNet.test(testData1, testData2)

    predicted = res > 0.5

    print "predicted"
    print predicted

    correct = (testLabels == predicted).sum() * 1.0 / len(res)

    print "params[fold]"
    print params[fold]

    print "correct"
    print correct
    correctForParams += [correct]

    fold += 1

  for i in xrange(len(params)):
    print "parameter tuple " + str(params[i]) + " achieved correctness of " + str(correctForParams[i])


def similarityEmotionsMain():
  trainData1, trainData2, trainLabels, testData1, testData2, testLabels =\
       splitSimilaritiesPIEEmotions(instanceToPairRatio=2, equalize=args.equalize)

  print "training with dataset of size ", len(trainData1)
  print len(trainData1)

  print "testing with dataset of size ", len(testData1)
  print len(testData1)

  if args.relu:
    if args.rmsprop:
      learningRate = 0.005
      rbmLearningRate = 0.01
      maxMomentum = 0.95
    else:
      learningRate = 0.001
      rbmLearningRate = 0.005
      maxMomentum = 0.95

    visibleActivationFunction = Identity()
    hiddenActivationFunction = RectifiedNoisy()
    # IMPORTANT: SCALE THE DATA IF YOU USE GAUSSIAN VISIBlE UNITS
    testData1 = scale(testData1)
    testData2 = scale(testData2)
    trainData1 = scale(trainData1)
    trainData2 = scale(trainData2)

  else:
    if args.rmsprop:
      learningRate = 0.001
      rbmLearningRate = 0.005
      maxMomentum = 0.95
    else:
      learningRate = 0.001
      rbmLearningRate = 0.005
      maxMomentum = 0.95

    visibleActivationFunction = Sigmoid()
    hiddenActivationFunction = Sigmoid()

  simNet = similarity.SimilarityNet(learningRate=learningRate,
                                    maxMomentum=maxMomentum,
                                    visibleActivationFunction=visibleActivationFunction,
                                    hiddenActivationFunction=hiddenActivationFunction,
                                    rbmNrVis=1200,
                                    rbmNrHid=args.nrHidden,
                                    rbmLearningRate=rbmLearningRate,
                                    rbmDropoutHid=1.0,
                                    rbmDropoutVis=1.0,
                                    rmsprop=False,
                                    trainingEpochsRBM=args.rbmepochs,
                                    nesterovRbm=True,
                                    sparsityConstraint=args.sparsity,
                                    sparsityRegularization=0.001,
                                    sparsityTraget=0.01)

  print "training with ", trainLabels.sum(), "positive examples"
  print "training with ", len(trainLabels) - trainLabels.sum(), "negative examples"

  print "testing with ", testLabels.sum(), "positive examples"
  print "testing with ", len(testLabels) - testLabels.sum(), "negative examples"

  final = []
  for i in xrange(len(trainData1)):

      if i > 6:
        break
      # Create 1 by 1 image
      res = np.vstack([trainData1[i].reshape(40,30), trainData2[i].reshape(40,30)])

      final += [res]

  final = np.hstack(final)
  plt.imshow(final, cmap=plt.cm.gray)
  plt.axis('off')
  plt.show()

  simNet.train(trainData1, trainData2, trainLabels, epochs=args.epochs)

  res = simNet.test(testData1, testData2)

  predicted = res > 0.5

  correct = (testLabels == predicted).sum() * 1.0 / len(res)

  confMatrix = confusion_matrix(testLabels, predicted)
  print confMatrix

  print correct

def similarityEmotionsSameSubject():
  trainData1, trainData2, trainLabels, testData1, testData2, testLabels =\
       splitEmotionsMultiPieKeepSubjectsTestTrain(instanceToPairRatio=2, equalize=args.equalize)

  print "training with dataset of size ", len(trainData1)
  print len(trainData1)

  print "testing with dataset of size ", len(testData1)
  print len(testData1)

  if args.relu:
    if args.rmsprop:
      learningRate = 0.001
      rbmLearningRate = 0.005
      maxMomentum = 0.95
    else:
      learningRate = 0.001
      rbmLearningRate = 0.005
      maxMomentum = 0.95

    visibleActivationFunction = Identity()
    hiddenActivationFunction = RectifiedNoisy()

    testData1 = scale(testData1)
    testData2 = scale(testData2)
    trainData1 = scale(trainData1)
    trainData2 = scale(trainData2)

  else:
    if args.rmsprop:
      learningRate = 0.001
      rbmLearningRate = 0.005
      maxMomentum = 0.95
    else:
      learningRate = 0.001
      rbmLearningRate = 0.005
      maxMomentum = 0.95

    visibleActivationFunction = Sigmoid()
    hiddenActivationFunction = Sigmoid()

  simNet = similarity.SimilarityNet(learningRate=learningRate,
                                    maxMomentum=maxMomentum,
                                    visibleActivationFunction=visibleActivationFunction,
                                    hiddenActivationFunction=hiddenActivationFunction,
                                    rbmNrVis=1200,
                                    rbmNrHid=args.nrHidden,
                                    rbmLearningRate=rbmLearningRate,
                                    rbmDropoutHid=1.0,
                                    rbmDropoutVis=1.0,
                                    rmsprop=False,
                                    trainingEpochsRBM=args.rbmepochs,
                                    nesterovRbm=True,
                                    sparsityConstraint=args.sparsity,
                                    sparsityRegularization=0.001,
                                    sparsityTraget=0.01)

  print "training with ", trainLabels.sum(), "positive examples"
  print "training with ", len(trainLabels) - trainLabels.sum(), "negative examples"

  print "testing with ", testLabels.sum(), "positive examples"
  print "testing with ", len(testLabels) - testLabels.sum(), "negative examples"


  simNet.train(trainData1, trainData2, trainLabels, epochs=args.epochs)

  res = simNet.test(testData1, testData2)

  predicted = res > 0.5

  correct = (testLabels == predicted).sum() * 1.0 / len(res)

  confMatrix = confusion_matrix(testLabels, predicted)
  print confMatrix

  print correct

def main():
  if args.cv:
    similarityCV()
  elif args.cvEmotion:
    similarityCVEmotions()
  elif args.diffsubjects:
    similarityDifferentSubjectsMain()
  elif args.testYaleMain:
    similarityMainTestYale()
  elif args.emotionsdiff:
    similarityEmotionsMain()
  elif args.emotionsdiffsamesubj:
    similarityEmotionsSameSubject()
  else:
    similarityMain()

if __name__ == '__main__':
  main()
