""" The aim of this module is to do face detection. This is required in order to

crop some of the input databases, because they are not centered"""

__author__ = "Mihaela Rosca"
__contact__ = "mihaela.c.rosca@gmail.com"

import cv2

# XML file with the recognition data
CASCADE_FN = "haarcascade_frontalface_default.xml"


"""Needs the image to already be black and white """
def cropFace(image, rescaleForReconigtion=2):
  cascade = cv2.CascadeClassifier(CASCADE_FN)
  imageScaled = cv2.resize(image, (image.shape[0] / rescaleForReconigtion ,
                            image.shape[1] / rescaleForReconigtion))

  # The image might already be equalized, so no need for that here
  gray = cv2.equalizeHist(imageScaled)
  rects = cascade.detectMultiScale(gray, 1.1, 3)

  # You need to find exactly one face in the picture
  print "len(rects)"
  print len(rects)
  if len(rects) is not 1:
    return None

  x, y, w, h = map(lambda x: x * rescaleForReconigtion,  rects[0])
  face = image[y:y + h, x:x + w]
  return face
