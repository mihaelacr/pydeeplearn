import sys
import cv2
from skimage.transform import resize
import numpy as np
import faceRecognition
import scipy
import matplotlib.pyplot as plt

# We need this to import other modules
sys.path.append("..")
from read import readfacedatabases
from lib import common

SMALL_SIZE = (40, 30)
SQUARE_SIZE = (48, 48)


def preprocess(image, faceCoordinates):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Step 1: crop the the image
  cropped = faceRecognition.cropFace(image, faceCoordinates)

  # Step 2: Resize
  resized = resize(cropped, SMALL_SIZE)

  # Step 3: Equalize the image (needs to be done in the same way it has been with the training data)
  equalized = readfacedatabases.equalizeFromFloatCLAHE(resized, SMALL_SIZE)
  return np.reshape(equalized, SMALL_SIZE)

def testImage(image, faceCoordinates, net):
  testImg = preprocess(image, faceCoordinates)

  # IMPORTANT: scale the image for it to be testable
  test = common.scale(testImg.reshape(1, len(testImg)))
  probs, emotion = net.classify(test)

  # classify returns a vector, as it is made to classify multiple test instances
  # at the same time.
  # We check if emotion is iterable before getting the first element, in case
  # someone uses an api in which a vector is not returned.
  if hasattr(emotion, '__iter__'):
    emotion = emotion[0]

  print "probs"
  print probs
  print "label"
  print emotion

  return emotion
