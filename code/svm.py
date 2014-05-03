from sklearn import svm

from common import *

def svmOnHiddenActivations(dbnNet, train, test, trainLabels, testLabels):
  classifier = svm.SVC()

  # trainHiddenRepresentations = dbnNet.getHiddenActivations(trainingScaledVectors)[-1]
  # classifier.fit(trainHiddenRepresentations, trainLabels)

  # testHiddenRepresentation = dbnNet.getHiddenActivations(testingScaledVectors)[-1]
  # predicted = classifier.predict(testHiddenRepresentation)

  # print getClassificationError(predicted, testLabels)

  trainHiddenRepresentations = dbnNet.hiddenActivations(train)

  trainHiddenRepresentations = scale(trainHiddenRepresentations)
  classifier.fit(trainHiddenRepresentations, trainLabels)

  testHiddenRepresentation = scale(testHiddenRepresentation)
  testHiddenRepresentation = dbnNet.hiddenActivations(test)
  predicted = classifier.predict(testHiddenRepresentation)

  print getClassificationError(predicted, testLabels)


def SVMCV():
  pass