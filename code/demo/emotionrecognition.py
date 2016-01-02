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

nrToEmotion = {
  0: "happy",
  1: "sad",
  2: "surprise"
}

def testImage(image, faceCoordinates, net, save=True):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  size  = SMALL_SIZE

  # Step 1: crop the the image
  cropped = faceRecognition.cropFace(image, faceCoordinates)

  # Step 2: Resize
  resized = resize(cropped, size)

  # Step 3: Equalize the image (needs to be done in the same way it has been with the training data)
  testImg = readfacedatabases.equalizeFromFloatCLAHE(resized, size)

  # Step4: Test the image with the network
  # IMPORTANT: scale the image for it to be testable
  test = common.scale(testImg.reshape(1, len(testImg)))
  probs, emotion = net.classify(test)

  print "probs"
  print probs
  print nrToEmotion[emotion[0]]

  return vals[0]
