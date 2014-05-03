from sklearn import svm, grid_search

from common import *

def svmOnHiddenActivations(dbnNet, train, test, trainLabels, testLabels):
  classifier = svm.SVC()

  # trainHiddenRepresentations = dbnNet.getHiddenActivations(trainingScaledVectors)[-1]
  # classifier.fit(trainHiddenRepresentations, trainLabels)

  # testHiddenRepresentation = dbnNet.getHiddenActivations(testingScaledVectors)[-1]
  # predicted = classifier.predict(testHiddenRepresentation)

  # print getClassificationError(predicted, testLabels)

  trainHiddenRepresentations = dbnNet.getHiddenActivations(train)

  trainHiddenRepresentations = scale(trainHiddenRepresentations)
  classifier.fit(trainHiddenRepresentations, trainLabels)

  testHiddenRepresentation = scale(testHiddenRepresentation)
  testHiddenRepresentation = dbnNet.getHiddenActivations(test)
  predicted = classifier.predict(testHiddenRepresentation)

  print getClassificationError(predicted, testLabels)


def SVMCV(dbnNet, train, trainLabels, test, testLabels):
  trainHiddenRepresentations = dbnNet.getHiddenActivations(train)
  trainHiddenRepresentations = scale(trainHiddenRepresentations)

  trainHiddenRepresentations = scale(trainHiddenRepresentations)
  parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
  classifier = svm.SVC()
  gridseach = grid_search.GridSearchCV(classifier, parameters)
  gridseach.fit(train, trainLabels)

  testHiddenRepresentation = dbnNet.getHiddenActivations(test)
  testHiddenRepresentation = scale(testHiddenRepresentation)
  predicted = gridseach.predict(testHiddenRepresentation)

  print getClassificationError(predicted, testLabels)
