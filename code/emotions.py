""" The aim of this file is to contain all the function
and the main which have to do with emotion recognition, especially
with the Kanade database."""

import glob

import DimensionalityReduction

"""
  Arguments:
    big: should the big or small images be used?
    folds: which folds should be used (1,..5) (a list). If None is passed all
    folds are used
"""
def deepBeliefKanade(big=False, folds=None):
  if big:
    files = glob.glob('kanade_150*.pickle')
  else:
    files = glob.glob('kanade_f*.pickle')

  if not folds:
    folds = range(1, 6)

  # Read the data from them. Sort out the files that do not have
  # the folds that we want
  # TODO: do this better (with regex in the file name)
  # DO not reply on the order returned
  files = files[folds]

  data = []
  labels = []
  for filename in files:
    with open(filename, "rb") as  f:
      # Sort out the labels from the data
      dataAndLabels = pickle.load(f)
      foldData = dataAndLabels[0:-1 ,:]
      foldLabels = dataAndLabels[-1,:]
      data.append(foldData)
      labels.append(foldLabels)

  # Do LDA

  # Create the network

  # Test

  # You can also group the emotions into positive and negative to see
  # if you can get better results (probably yes)
  pass