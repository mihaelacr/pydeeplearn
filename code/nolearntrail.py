from nolearn.dbn import DBN
from readfacedatabases import *
from sklearn import cross_validation
from sklearn.metrics import zero_one_score
from sklearn.metrics import classification_report

import argparse
import numpy as np

from common import *

parser = argparse.ArgumentParser(description='nolearn test')
parser.add_argument('--equalize',dest='equalize',action='store_true', default=False,
                    help="if true, the input images are equalized before being fed into the net")
parser.add_argument('--maxEpochs', type=int, default=1000,
                    help='the maximum number of supervised epochs')

args = parser.parse_args()


def KanadeClassifier():

  clf = DBN(
      [1200, 1500, 1500, 1500, 7],
      learn_rates=0.01,
      learn_rates_pretrain=0.05,
      learn_rate_decays=0.9,
      use_re_lu=True,
      nesterov=True,
      momentum=0.95,
      # dropouts=[0.8, 0.5, 0.5, 0.5],
      real_valued_vis=True,
      minibatch_size=20,
      epochs=args.maxEpochs,
      verbose=False
      )

  data, labels = readKanade(False, None, equalize=args.equalize)

  data = scale(data)

  data, labels = shuffle(data, labels)

  labels = np.argmax(labels, axis=1)

  # Split data for training and testing
  kf = cross_validation.KFold(n=len(data), n_folds=5)
  for train, test in kf:
    break

  trainData = data[train]
  trainLabels = labels[train]


  testData = data[test]
  testLabels = labels[test]


  clf.fit(trainData, trainLabels)

  predictedLabels = clf.predict(testData)

  print "testLabels"
  print testLabels
  print predictedLabels

  print "Accuracy:", zero_one_score(testLabels, predictedLabels)
  print "Classification report:"
  print classification_report(testLabels, predictedLabels)

if __name__ == '__main__':
  KanadeClassifier()