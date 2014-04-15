""" The aim of this module is to do face detection. This is required in order to

crop some of the input databases, because they are not centered"""

import cv2
import numpy as np

# Create window for image display
CASCADE_FN = "haarcascade_frontalface_default.xml"

# The scale used for face recognition.
# It is important as the face recognition algorithm works better on small images
# Also helps with removing faces that are too far away
RESIZE_SCALE = 3
RECTANGE_COLOUR = (255, 0, 0)
THICKNESS = 2


# TODO: add the rescaling
"""Needs the image to already be black and white """
def cropFace(image):
  print "image.shape"
  print image.shape
  cascade = cv2.CascadeClassifier(CASCADE_FN)
  imageScaled = cv2.resize(image, (image.shape[0] / RESIZE_SCALE ,
                            image.shape[1] / RESIZE_SCALE))


  gray = cv2.equalizeHist(imageScaled)
  rects = cascade.detectMultiScale(gray, 1.3, 5)
  print "rects"
  print rects
  # You need to find exactly one face in the picture
  assert len(rects) == 1

  x, y, w, h = map(lambda x: x * RESIZE_SCALE,  rects[0])
  face = image[y:y + h, x:x + w]
  return face
