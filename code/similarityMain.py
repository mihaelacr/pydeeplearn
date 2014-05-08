import argparse

from similarity_utils import *
import similarity

parser = argparse.ArgumentParser(description='digit recognition')
parser.add_argument('--relu', dest='relu',action='store_true', default=False,
                    help=("if true, trains the RBM or DBN with a rectified linear unit"))

args = parser.parse_args()


def main():
  trainData1, trainData2, testData1, testData2, similaritiesTrain, similaritiesTest = splitData(10)

  simNet = similarity.SimilarityNet(0.01, learningRate=learningRate,
                                    binary=True,
                                    rbmNrVis=1200,
                                    rbmNrHid=500,
                                    rbmLearningRate=0.001,
                                    rbmDropoutHid=1.0,
                                    rbmDropoutVis=1.0)

  simNet.train(trainData1, trainData2, similaritiesTrain)

  res = simNet.test(testData1, testData2)
  print res


if __name__ == '__main__':
  main()
