from skimage.transform import resize
import cv2
import numpy as np
import sys

import face_detection

# We need this to import other modules
sys.path.append("..")
from read import readfacedatabases
from lib import common

SMALL_SIZE = (40, 30)
SQUARE_SIZE = (48, 48)


def preprocess(image, faceCoordinates, return_vector=False):
  """Preprocess the input image according to the face coordinates detected
   by a face recognition engine.

   This method:
     * crops the input image, keeping only the face given by faceCoordinates
     * transforms the picture into black and white
     * equalizes the input image

   If return_vector is True, returns a vector by concatenating the rows of the
   processed image. Otherwise, a matrix (2-d numpy array) is returned.

   This method needs to be called both for training and testing.
  """
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Step 1: crop the the image
  cropped = face_detection.cropFace(image, faceCoordinates)

  # Step 2: Resize
  resized = np.ascontiguousarray(resize(cropped, SMALL_SIZE))

  # Step 3: Equalize the image (needs to be done in the same way it has been with the training data)
  equalized = readfacedatabases.equalizeFromFloatCLAHE(resized, SMALL_SIZE)
  if return_vector:
    return equalized
  return np.reshape(equalized, SMALL_SIZE)

def testImage(image, faceCoordinates, emotion_classifier):
  """Classifies the emotions in the input image according to the face coordinates
  detected by a face detection engine.

  First calls preprocess and then uses the given emotion_classifier to detect
  emotions in the processed image.

  """
  testImg = preprocess(image, faceCoordinates, return_vector=True)

  # IMPORTANT: scale the image for it to be testable
  test = common.scale(testImg.reshape(1, len(testImg)))
  probs, emotion = emotion_classifier.classify(test)

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
