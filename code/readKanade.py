import numpy as np
import glob
import cPickle as pickle
import matplotlib.pyplot as plt

# Open the files that are from the kanade database and
# parse them in order to get the emotions

# TODO: write them as tuples not arrays
def readTxtWritePickle(filename):
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

  with open(pickleFileName,"wb") as f:
    pickle.dump(lines, f)


def main():
  # Read all the given files, parse them and write them as a
  # numpy array using cpikle
  files = glob.glob('kanade*.txt')
  for f in files:
    readTxtWritePickle(f)

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
  face = np.array(face).reshape(resizeShape).T
  plt.imshow(face, cmap=plt.cm.gray)
  plt.show()

if __name__ == '__main__':
  viewTestImage(big=True)





