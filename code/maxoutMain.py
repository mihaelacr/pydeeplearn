from __future__ import division
import os

import numpy as np
import argparse

from pylearn2.train import Train
from pylearn2.datasets.mnist import MNIST
from pylearn2.models import mlp, maxout
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import MonitorBased
from pylearn2.train_extensions import best_params
from pylearn2.utils import serial
from pylearn2.costs.mlp.dropout import Dropout
from theano import function
from theano import tensor as T
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

from readfacedatabases import *
from common import *

parser = argparse.ArgumentParser(description='digit recognition')
parser.add_argument('--mnist', dest='mnist',action='store_true', default=False,
                    help=("if true, trains the net on MNIST data"))
parser.add_argument('--pie',dest='pie',action='store_true', default=False,
                    help="if true, trains the net on MultiPIE data")
args = parser.parse_args()

class MultiPIE(DenseDesignMatrix):

  def __init__(self, name, indices=None):
    self.name = name

    x, y = readMultiPIE(vectorLabels=False)
    x = x[indices]
    y = y[indices]

    self.label_names = ['Neutral','Surprise','Squint','Smile','Disgust','Scream']
    self.n_classes = len(self.label_names)

    self.label_map = {k: v for k, v in zip(self.label_names, range(self.n_classes))}
    self.label_unmap = {v: k for k, v in zip(self.label_names, range(self.n_classes))}

    axes=('c', 0, 1, 'b')

    super(MultiPIE, self).__init__(y=y, x=x, axes=axes)

  def convert(self, x):
    return self.label_map[x]

  def unconvert(self, x):
    return self.label_unmap[x]


def MNISTmain():
  # TODO: max_col_norm
  h0 = maxout.Maxout(layer_name='h0', num_units=1200, num_pieces=2, W_lr_scale=1.0, irange=0.005, b_lr_scale=1.0)
  h1 = maxout.Maxout(layer_name='h1', num_units=1200, num_pieces=2, W_lr_scale=1.0, irange=0.005, b_lr_scale=1.0)
  # h2 = maxout.Maxout(layer_name='h2, num_units=1200, num_pieces=2, W_lr_scale=1.0, irange=0.005, b_lr_scale=1.0)
  outlayer = mlp.Softmax(layer_name='y', n_classes=10, irange=0)

  # layers = [h0, h1, h2, outlayer]
  layers = [h0, h1, outlayer]

  model = mlp.MLP(layers, nvis=784)
  train = MNIST('train', one_hot=1, start=0, stop=50000)
  valid = MNIST('train', one_hot=1, start=50000, stop=60000)
  test = MNIST('test', one_hot=1, start=0, stop=10000)

  monitoring = dict(valid=valid)
  termination = MonitorBased(channel_name="valid_y_misclass", N=100)
  extensions = [best_params.MonitorBasedSaveBest(channel_name="valid_y_misclass",
                                                 save_path="train_best.pkl")]

  algorithm = sgd.SGD(0.1, batch_size=100, cost=Dropout(),
                      monitoring_dataset = monitoring, termination_criterion = termination)

  save_path = "train_best.pkl"

  if os.path.exists(save_path):
      model = serial.load(save_path)
  else:
    print 'Running training'
    train_job = Train(train, model, algorithm, extensions=extensions, save_path="train.pkl", save_freq=1)
    train_job.main_loop()

  X = model.get_input_space().make_batch_theano()
  Y = model.fprop(X)

  y = T.argmax(Y, axis=1)
  f = function([X], y)
  yhat = f(test.X)

  y = np.squeeze(test.get_targets())

  print 'accuracy', (y==yhat).sum() / y.size

def MultiPIE():
  h0 = maxout.Maxout(layer_name='h0', num_units=1200, num_pieces=2, W_lr_scale=1.0, irange=0.005, b_lr_scale=1.0)
  h1 = maxout.Maxout(layer_name='h1', num_units=1500, num_pieces=2, W_lr_scale=1.0, irange=0.005, b_lr_scale=1.0)
  h2 = maxout.Maxout(layer_name='h2', num_units=1500, num_pieces=2, W_lr_scale=1.0, irange=0.005, b_lr_scale=1.0)
  outlayer = mlp.Softmax(layer_name='y', n_classes=6, irange=0)

  layers = [h0, h1, h2, outlayer]

  model = mlp.MLP(layers, nvis=1200)

  trainIndices, validationIndices, testIndices = getMultiPIEindices()
  train = MultiPIE('train', indices=trainIndices)
  valid = MultiPIE('valid', indices=validationIndices)
  test = MultiPIE('test',   indices=testIndices)

  monitoring = dict(valid=valid)
  termination = MonitorBased(channel_name="valid_y_misclass", N=100)
  extensions = [best_params.MonitorBasedSaveBest(channel_name="valid_y_misclass",
                                                 save_path="train_best.pkl")]

  algorithm = sgd.SGD(0.1, batch_size=100, cost=Dropout(),
                      monitoring_dataset=monitoring, termination_criterion=termination)

  save_path = "train_best_pie.pkl"

  train_job = Train(train, model, algorithm, extensions=extensions, save_path="trainpie.pkl", save_freq=1)
  train_job.main_loop()

  X = model.get_input_space().make_batch_theano()
  Y = model.fprop(X)

  y = T.argmax(Y, axis=1)
  f = function([X], y)
  yhat = f(test.X)

  y = np.where(test.get_targets())[1]
  print 'accuracy', (y==yhat).sum() / y.size

def getMultiPIEindices():
  x, y = readMultiPIE()
  l = len(x)

  kf = cross_validation.KFold(n=l, n_folds=5)
  for train, test in kf:
    break

  allIndices = np.random.permutation(l)

  testIndices = allIndices[test]
  stopValidation = len(train)/10
  validationIndices = allIndices[train][0: stopValidation]
  trainIndices = allIndices[train][stopValidation: ]

  return trainIndices, validationIndices, testIndices

if __name__ == '__main__':
  if args.mnist:
    MNISTmain()
  if args.pie:
    MultiPIE()