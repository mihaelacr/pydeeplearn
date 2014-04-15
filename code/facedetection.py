""" The aim of this module is to do face detection. This is required in order to

crop some of the input databases, because they are not centered"""

import cv2
import numpy as np

# Create window for image display
CASCADE_FN = "haarcascade_frontalface_default.xml"


"""Needs the image to already be black and white """
def cropFace(image, rescaleForReconigtion=2):
  cascade = cv2.CascadeClassifier(CASCADE_FN)
  imageScaled = cv2.resize(image, (image.shape[0] / rescaleForReconigtion ,
                            image.shape[1] / rescaleForReconigtion))


  gray = cv2.equalizeHist(imageScaled)
  rects = cascade.detectMultiScale(gray, 1.2, 5)
  # You need to find exactly one face in the picture
  # assert len(rects) == 1
  print "len(rects)"
  print len(rects)
  if len(rects) == 0:
    return None
  # if len(rects) is not 1:

  # assert len(rects) == 1

  x, y, w, h = map(lambda x: x * rescaleForReconigtion,  rects[0])
  face = image[y:y + h, x:x + w]
  return face
