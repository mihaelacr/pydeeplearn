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


def getFaceCoordinates(image):
  """Uses openCV to detect the face present in the input image.

  Returns a list of length 4, with the two corners of the rectangle that define
  the position of the face: [x1, y1, x2, y2], where (x1, y1) and (x2, y2)
  are the defining corners of the rectangle.
  """

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
  corners = [r[0], r[1], r[0] + r[2], r[1] + r[3]]

  return map((lambda x: RESIZE_SCALE * x), corners)


# Turns an image into a rbg image by replicating the 2d data on each of the axis.
def to_rgb1(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

def drawFace(image, faceCoordinates, emotion, emotion_to_text, emotion_to_image=None):
  """ Draws the face information (together with emotion information) on the input image. """

  # Draw the face detection rectangles.
  cv2.rectangle(np.asarray(image),
                (faceCoordinates[0], faceCoordinates[1]),
                (faceCoordinates[2], faceCoordinates[3]),
                RECTANGE_COLOUR,
                thickness=THICKNESS)

  # Display the emotion on the webcam stream.
  if emotion is not None:
    cv2.putText(image,
                # Get the text associated with this emotion, but
                # if we do not have one just display the integer.
                emotion_to_text.get(emotion, ""),
                (faceCoordinates[0], faceCoordinates[2]),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                BOX_COLOR,
                thickness=2)

    # Add a nice smiley to show the classification
    if emotion_to_image:
      smallImage = emotion_to_image[emotion]
      smallImage = cv2.resize(smallImage, (faceCoordinates[0], faceCoordinates[1]))
      smallImage = to_rgb1(smallImage)
      if smallImage.shape[0] > image.shape[0] or smallImage.shape[1] > image.shape[0]:
        return
      image[0:0+smallImage.shape[0], 0:0 + smallImage.shape[1]] = smallImage


def cropFace(image, faceCoordinates):
  return image[faceCoordinates[1]: faceCoordinates[3],
                faceCoordinates[0]: faceCoordinates[2]]
