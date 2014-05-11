from sklearn import cross_validation
from readfacedatabases import *


# TODO: move to common?
def splitTrainTest(data1, data2, labels1, labels2, ratio):
  assert len(data1) == len(data2)
  assert len(labels1) == len(labels2)
  assert len(labels1) == len(data1)
  # Random data for training and testing
  kf = cross_validation.KFold(n=len(data1), n_folds=ratio)
  for train, test in kf:
    break

  return (data1[train], data1[test], data2[train], data2[test],
          labels1[train], labels1[test], labels2[train], labels2[test])

def splitShuffling(shuffling, labelsShuffling):
  shuffledData1 = shuffling[0: len(shuffling) / 2]
  shuffledData2 = shuffling[len(shuffling)/2 :]

  labelsData1 = labelsShuffling[0: len(shuffling) /2]
  labelsData2 = labelsShuffling[len(shuffling)/2:]

  shuffledData1 = np.array(shuffledData1)
  shuffledData1 = np.array(shuffledData2)
  labelsData1 = np.array(labelsData1)
  labelsData2 = np.array(labelsData2)

  return shuffledData1, shuffledData2, labelsData1, labelsData2

# TODO: I think this can be written easier with the code similar to the Emotions one
# you can create more tuples than just one per image
# you can put each image in 5 tuples and that will probably owrk better
# it might be useful to also give the same image twice
def splitDataMultiPIESubject(imgsPerSubject=None):
  subjectsToImgs = readMultiPIESubjects()

  data1, data2, subjects1, subjects2, shuffling, subjectsShuffling = splitDataInPairsWithLabels(subjectsToImgs, imgsPerSubject, None)

  trainData1, testData1, trainData2, testData2, trainSubjects1, testSubjects1,\
        trainSubjects2, testSubjects2 = splitTrainTest(data1, data2, subjects1, subjects2, 5)

  print "trainData1.shape"
  print trainData1.shape

  shuffledData1, shuffledData2, subjectsData1, subjectsData2 = splitShuffling(shuffling, subjectsShuffling)

  print len(shuffledData1)
  print len(shuffledData2)

  trainShuffedData1, testShuffedData1, trainShuffedData2, testShuffedData2,\
    trainShuffledSubjects1, testShuffledSubjects1, trainShuffledSubjects2, testShuffledSubjects2 =\
        splitTrainTest(shuffledData1, shuffledData2,
                      subjectsData1, subjectsData2, 5)

  trainData1 = np.vstack((trainData1, trainShuffedData1))

  print "trainData1.shape"
  print trainData1.shape

  trainData2 = np.vstack((trainData2, trainShuffedData2))

  testData1 = np.vstack((testData1, testShuffedData1))
  testData2 = np.vstack((testData2, testShuffedData2))

  trainSubjects1 = np.hstack((trainSubjects1, trainShuffledSubjects1))
  testSubjects1 = np.hstack((testSubjects1, testShuffledSubjects1))

  trainSubjects2 = np.hstack((trainSubjects2, trainShuffledSubjects2))
  testSubjects2 = np.hstack((testSubjects2, testShuffledSubjects2))

  assert len(subjects1) == len(subjects2)
  assert len(trainSubjects1) == len(trainSubjects1)
  assert len(testSubjects1) == len(testSubjects2)

  similaritiesTrain = similarityDifferentLabels(trainSubjects1, trainSubjects2)
  similaritiesTest = similarityDifferentLabels(testSubjects1, testSubjects2)

  print "trainSubjects1.shape"
  print trainSubjects1.shape

  print "similaritiesTrain.shape"
  print similaritiesTrain.shape
  print similaritiesTrain

  assert len(trainData1) == len(trainData2)
  assert len(testData1) == len(testData2)

  trainData1, trainData2, similaritiesTrain = shuffle3(trainData1, trainData2, similaritiesTrain)
  testData1, testData2, similaritiesTest = shuffle3(testData1, testData2, similaritiesTest)

  return trainData1, trainData2, testData1, testData2, similaritiesTrain, similaritiesTest


def splitDataInPairsWithLabels(labelsToImages, imgsPerLabel, labelsToTake=None):
  data1 = []
  data2 = []

  shuffling = []
  labelsShuffling = []
  labels1 = []
  labels2 = []

  for label, images in labelsToImages.iteritems():
    if labelsToTake is not None and label not in labelsToTake:
      print "skipping subject"
      continue

    # The database might contain the labels in similar
    # poses, and illumination conditions, so shuffle before
    np.random.shuffle(images)

    if imgsPerLabel is not None:
      images = images[:imgsPerLabel]

    delta = len(images) / 4 + label % 2
    last2Index = 2 *delta
    data1 += images[0: delta]
    data2 += images[delta: last2Index]

    labels1 += [label] * delta
    labels2 += [label] * delta

    imagesForShuffling = images[last2Index : ]
    shuffling += imagesForShuffling
    labelsShuffling += [label] * len(imagesForShuffling)

  print "len(labelsShuffling)"
  print len(labelsShuffling)

  print "shuffling"
  print len(shuffling)

  assert len(shuffling) == len(labelsShuffling)
  shuffling, labelsShuffling = shuffleList(shuffling, labelsShuffling)

  print len(data1)
  print len(data2)
  assert len(data1) == len(data2)

  data1 = np.array(data1)
  data2 = np.array(data2)
  labels1 = np.array(labels1)
  labels2 = np.array(labels2)
  shuffling = np.array(shuffling)
  labelsShuffling = np.array(labelsShuffling)

  return data1, data2, labels1, labels2, shuffling, labelsShuffling

def splitDataAccordingToLabels(labelsToImages, labels, imgsPerLabel=None):
  data1, data2, labels1, labels2, shuffling, labelsShuffling  = splitDataInPairsWithLabels(labelsToImages, imgsPerLabel, labelsToTake=labels)

  shuffledData1, shuffledData2, labelsData1, labelsData2 = splitShuffling(shuffling, labelsShuffling)

  data1 = np.vstack((data1, shuffledData1))
  data2 = np.vstack((data2, shuffledData2))

  labels1 = np.hstack((labels1, labelsData1))
  labels2 = np.hstack((labels2, labelsData2))

  return data1, data2, labels1, labels2

def similarityDifferentLabels(labels1, labels2):
  assert len(labels1) == len(labels2)
  return labels1 == labels2

def splitSimilarityYale():
  subjectsToImgs = readCroppedYaleSubjects()

  # Get all subjects
  data1, data2, subjects1, subjects2 = splitDataAccordingToLabels(subjectsToImgs,
                                          None, imgsPerSubject=None)

  return data1, data2, similarityDifferentLabels(subjects1, subjects2)

def splitSimilaritiesPIEEmotions():
  emotionToImages = readMultiPIEEmotions()
  # Get all emotions
  data1, data2, emotions1, emotions2 = splitDataAccordingToLabels(emotionToImages,
                                          None, imgsPerSubject=None)
  kf = cross_validation.KFold(n=len(data1), n_folds=ratio)
  for train, test in kf:
    break

  labels = similarityDifferentLabels(emotions1, emotions2)

  return (data1[train], data2[train], labels[train],
          data1[test], data2[test], labels[test])
