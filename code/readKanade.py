import numpy as np
import glob
import cPickle as pickle
import matplotlib.pyplot as plt


SMALL_SIZE = (40, 30)
BIG_SIZE = (200, 150)

# Open the files that are from the kanade database and
# parse them in order to get the emotions

# TODO: write them as tuples not arrays
def readTxtWritePickle(filename, big=False):
  if big:
    resizeShape = (150, 200)
  else:
    resizeShape = (30, 40)

  with open(filename) as f:
    # TODO: replace this with a functiOn call from the library
    lines = []
    for l in f:
      numbers = map(lambda x: float(x), l.split("\t"))
      lines.append(numbers)

    lines = np.array(lines)

  print lines.shape

  # Create the new name for the file
  pickleFileName = filename[0:-3] + "pickle"

  lines = lines.T

  data = lines[:, 0:-1]
  print "data.shape"
  print data.shape
  print "data[0].shape"
  print data[0].shape

  # Processing of the data
  reshapeF = lambda x: x.reshape(resizeShape).T.reshape(-1)
  reshapeF(data[0])
  data = np.array(map(reshapeF, data))

  lines[:, 0:-1] = data

  with open(pickleFileName,"wb") as f:
    pickle.dump(lines, f)


def main():
  # Read all the given files, parse them and write them as a
  # numpy array using cpikle

  # The small files
  files = glob.glob('kanade_f*.txt')
  for f in files:
    readTxtWritePickle(f, big=False)

  # The big files
  files = glob.glob('kanade_150*.txt')
  for f in files:
    readTxtWritePickle(f, big=True)

# Method used for testing images
def viewTestImage(big=False):
  if big:
    filename = "kanade_150x200_fold_1.txt"
    resizeShape = (150, 200)
  else:
    filename = "kanade_fold_1.txt"
    resizeShape = (30, 40)

  with open(filename) as f:
    # TODO: replace this with a functiOn call from the library
    lines = []
    for l in f:
      numbers = map(lambda x: float(x), l.split("\t"))
      lines.append(numbers)

    lines = np.array(lines)

  # Let's see the image to check that it looks like a face
  face = lines[:, 0][0:-1]
  print face
  emotion = lines[:, 0][-1]
  print "emotion"
  print emotion
  face = np.array(face).reshape(resizeShape)
  print face.shape
  plt.imshow(face, cmap=plt.cm.gray)
  plt.show()

if __name__ == '__main__':
  main()
  viewTestImage(big=False)
  # main()





