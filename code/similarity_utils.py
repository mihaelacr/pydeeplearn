from sklearn import cross_validation
from readfacedatabases import *


# TODO: move to common?
def splitTrainTest(data1, data2, labels1, labels2, ratio):
  assert len(data1) == len(data2)
  assert len(labels1) == len(labels2)
  assert len(labels1) == len(data1)
  # Random data for training and testing
  kf = cross_validation.KFold(n=len(data1), k=ratio)
  for train, test in kf:
    break

  return (data1[train], data1[test], data2[train], data2[test],
          labels1[train], labels1[test], labels2[train], labels2[test])


# you can create more tuples than just one per image
# you can put each image in 5 tuples and that will probably owrk better
# it might be useful to also give the same image twice
def splitData(imgsPerSubject=None):
  subjectsToImgs = readMultiPIESubjects()

  data1, data2, subjects1, subjects2, shuffling, subjectsShuffling = splitSubjectData(subjectsToImgs, imgsPerSubject)

  trainData1, testData1, trainData2, testData2, trainSubjects1, testSubjects1,\
        trainSubjects2, testSubjects2 = splitTrainTest(data1, data2, subjects1, subjects2, 5)

  print "trainData1.shape"
  print trainData1.shape

  shuffledData1 = shuffling[0: len(shuffling) / 2]
  shuffledData2 = shuffling[len(shuffling)/2 :]

  subjectsData1 = subjectsShuffling[0: len(shuffling) /2]
  subjectsData2 = subjectsShuffling[len(shuffling)/2:]

  shuffledData2 = shuffledData2[:-1]
  subjectsData2 = subjectsData2[:-1]

  shuffledData1 = np.array(shuffledData1)
  shuffledData2 = np.array(shuffledData2)
  subjectsData1 = np.array(subjectsData1)
  subjectsData2 = np.array(subjectsData2)

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
  similaritiesTrain = (trainSubjects1 == trainSubjects2)
  similaritiesTest = (testSubjects1 == testSubjects2)


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


def splitSubjectData(subjectsToImgs, imgsPerSubject):

  data1 = []

  data2 = []

  shuffling = []
  subjectsShuffling = []
  subjects1 = []
  subjects2 = []

  for subject, images in subjectsToImgs.iteritems():
    if imgsPerSubject is not None:
      images = images[:imgsPerSubject]

    lastIndex = len(images)/ 4 + subject % 2
    delta = len(images)/ 4 + (subject + 1) % 2
    last2Index = lastIndex + delta
    data1 += images[0: lastIndex]
    data2 += images[lastIndex: last2Index]

    subjects1 += [subject] * lastIndex
    subjects2 += [subject] * delta

    imagesForShuffling = images[last2Index : ]
    shuffling += imagesForShuffling
    subjectsShuffling += [subject] * len(imagesForShuffling)

  print "len(subjectsShuffling)"
  print len(subjectsShuffling)

  print "shuffling"
  print len(shuffling)

  assert len(shuffling) == len(subjectsShuffling)
  shuffling, subjectsShuffling = shuffleList(shuffling, subjectsShuffling)

  print len(data1)
  print len(data2)

  # Warning: hack; now with the new subject thing it will not work
  data2 = data2[:-1]
  subjects2 = subjects2[:-1]

  data1 = np.array(data1)
  data2 = np.array(data2)
  subjects1 = np.array(subjects1)
  subjects2 = np.array(subjects2)

  return data1, data2, subjects1, subjects2, shuffling, subjectsShuffling
