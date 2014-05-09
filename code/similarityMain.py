import argparse
from sklearn import cross_validation

from similarity_utils import *
import similarity

parser = argparse.ArgumentParser(description='digit recognition')
parser.add_argument('--relu', dest='relu',action='store_true', default=False,
                    help=("if true, trains the RBM or DBN with a rectified linear unit"))
parser.add_argument('--cv', dest='cv',action='store_true', default=False,
                    help=("if true, does cv"))


args = parser.parse_args()


def similarityMain():
  trainData1, trainData2, testData1, testData2, similaritiesTrain, similaritiesTest = splitData(10)

  simNet = similarity.SimilarityNet(learningRate=0.1,
                                    maxMomentum=0.95,
                                    binary=True,
                                    rbmNrVis=1200,
                                    rbmNrHid=1000,
                                    rbmLearningRate=0.001,
                                    rbmDropoutHid=1.0,
                                    rbmDropoutVis=1.0)

  simNet.train(trainData1, trainData2, similaritiesTrain)

  res = simNet.test(testData1, testData2)

  error = (similaritiesTest - res)  * 1.0 / len(res)

  print res

  print error


def similarityCV():
  trainData1, trainData2, testData1, testData2, similaritiesTrain, similaritiesTest = splitData()

  params = [(0.0001, 0.01), (0.0001, 0.005), (0.001, 0.01), (0.001, 0.005)]
  kf = cross_validation.KFold(n=len(trainData1), k=len(params))

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

    res = simNet.test(testData1, testData2)

    # error = (similaritiesTest - res)  * 1.0 / len(res)

    predicted = res > 0.7

    error = (similaritiesTest == predicted).sum() * 1.0 / len(res)


    print res
    print error

    fold += 1


def main():
  if args.cv:
    similarityCV()
  else:
    similarityMain()

if __name__ == '__main__':
  main()
