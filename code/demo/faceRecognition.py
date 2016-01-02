import cv2
import numpy as np

# Create window for image display
CASCADE_FN = "haarcascade_frontalface_default.xml"

# The scale used for face recognition.
# It is important as the face recognition algorithm works better on small images
# Also helps with removing faces that are too far away
RESIZE_SCALE = 3
RECTANGE_COLOUR = (117, 30, 104)
BOX_COLOR = (255, 255, 255)
THICKNESS = 2

# Person by Catherine Please from The Noun Project
HAPPY_IMAGE = cv2.imread("icon_4895withoutalpha.png", cv2.IMREAD_GRAYSCALE)
# Sad by Cengiz SARI from The Noun Project
SAD_IMAGE = cv2.imread("icon_39345withoutalpha.png", cv2.IMREAD_GRAYSCALE)
# Surprise designed by Chris McDonnell from the thenounproject.com
SUPRISED_IMAGE = cv2.imread("icon_6231withoutalpha.png", cv2.IMREAD_GRAYSCALE)

EMOTIONS = [0, 1, 2]

EMOTION_TO_IMAGE = {
  0: HAPPY_IMAGE,
  1: SAD_IMAGE,
  2: SUPRISED_IMAGE
}

EMOTION_TO_TEXT = {
  0: "HAPPY",
  1: "SAD",
  2: "SUPRISE"
}

def getFaceCoordinates(image):
  cascade = cv2.CascadeClassifier(CASCADE_FN)
  img_copy = cv2.resize(image, (image.shape[1]/RESIZE_SCALE,
                                image.shape[0]/RESIZE_SCALE))
  gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
  gray = cv2.equalizeHist(gray)
  rects = cascade.detectMultiScale(gray, 1.2, 3)

  # If there is no face or if we have more than 2 faces return None
  # because we do not deal with that yet
  if len(rects) != 1:
    return None

  r = rects[0]
  new_r = map((lambda x: RESIZE_SCALE * x), r)
  return new_r


def to_rgb1(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

# (294, 454, 3) this is the shape of the frame
def drawFace(image, faceCoordinates, emotion):
  x = faceCoordinates[0]
  y = faceCoordinates[1]
  w = faceCoordinates[2]
  h = faceCoordinates[3]

  # Draw the face detection rectangles.
  cv2.rectangle(np.asarray(image), (x,y), (x + w, y + h), RECTANGE_COLOUR,
                  thickness=THICKNESS)

  #  Draw the emotion specifc emoticon.
  if emotion is not None:
    if emotion not in EMOTIONS:
      raise Exception("unknown emotion")
    else:
      cv2.putText(image, EMOTION_TO_TEXT[emotion], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, BOX_COLOR, thickness=2)
      smallImage = EMOTION_TO_IMAGE[emotion]

    smallImage = cv2.resize(smallImage, (x,y))
    smallImage = to_rgb1(smallImage)
    image[0:0+smallImage.shape[0], 0:0+smallImage.shape[1]] = smallImage

def cropFace(image, faceCoordinates):
  x = faceCoordinates[0]
  y = faceCoordinates[1]
  w = faceCoordinates[2]
  h = faceCoordinates[3]

  return image[y:y+h, x:x+w]


def getAndDrawFace(image, display=False):
  faceCoordinates = getFaceCoordinates(image)

  # If there is no face, do not continue
  if faceCoordinates is None:
    return None

  if display:
      drawFace(image, faceCoordinates)

  return cropFace(image, faceCoordinates)
