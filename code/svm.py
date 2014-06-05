from sklearn import svm, grid_search

from common import *

def svmOnHiddenActivations(dbnNet, train, test, trainLabels, testLabels):
  classifier = svm.SVC()

  # trainHiddenRepresentations = dbnNet.getHiddenActivations(trainingScaledVectors)[-1]
  # classifier.fit(trainHiddenRepresentations, trainLabels)

  # testHiddenRepresentation = dbnNet.getHiddenActivations(testingScaledVectors)[-1]
  # predicted = classifier.predict(testHiddenRepresentation)

  # print getClassificationError(predicted, testLabels)

  trainHiddenRepresentations = dbnNet.getHiddenActivations(train)[-1]

  trainHiddenRepresentations = scale(trainHiddenRepresentations)
  classifier.fit(trainHiddenRepresentations, trainLabels)

  testHiddenRepresentation = scale(testHiddenRepresentation)
  testHiddenRepresentation = dbnNet.getHiddenActivations(test)[-1]
  predicted = classifier.predict(testHiddenRepresentation)

  print getClassificationError(predicted, testLabels)


def SVMCV(dbnNet, train, trainLabels, test, testLabels):
  trainHiddenRepresentations = dbnNet.getHiddenActivations(train)[-1]
  # trainHiddenRepresentations = scale(trainHiddenRepresentations)

  testHiddenRepresentation = dbnNet.getHiddenActivations(test)[-1]
  # testHiddenRepresentation = scale(testHiddenRepresentation)

  parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
  classifier = svm.LinearSVC()
  gridseach = grid_search.GridSearchCV(classifier, parameters)
  gridseach.fit(trainHiddenRepresentations, trainLabels)

  print "params"
  print gridseach.get_params()
  predicted = gridseach.predict(testHiddenRepresentation)

  print getClassificationError(predicted, testLabels)
