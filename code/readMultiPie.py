"""The aim of this script is to read the multi pie dataset """

import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# TODO: make some general things with the path in order to make it work easily between
# lab machine and local
mat = scipy.io.loadmat('/home/aela/uni/project/Multi-PIE_Aligned/A_MultiPIE.mat')
data = mat['a_multipie']

def readMultiPie(data, show=False):
  # For all the subjects
  imgs = []
  labels = []
  for subject in xrange(147):
    for pose in xrange(5):
      for expression in xrange(6):
        for illumination in xrange(5):
            image = np.squeeze(data[subject,pose,expression,illumination,:])
            image = image.reshape(30,40).T
            imgs += [image]
            labels += [expression]
            if show:
              plt.imshow(image)
              plt.show()
  return imgs, labels

readMultiPie(data, show=True)




