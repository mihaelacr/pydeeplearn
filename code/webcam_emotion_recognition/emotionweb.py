import argparse
import cv2
import signal
import sys
import time

import faceRecognition
import ignoreoutput
import emotionrecognition
import cPickle as pickle

# We need this to import other modules
sys.path.append("..")

from lib import deepbelief

WINDOW_NAME = "Emotion recognition"
TIME_BETWEEN_FACE_CHECKS = 0.1

parser = argparse.ArgumentParser(description=("Live emotion recognition from the webcam"))
parser.add_argument('--displayWebcam', action='store_const', const=True,
                    help="determies if the image from the webcam is displayed")
parser.add_argument("--seeFaces", action='store_const', const=True,
                    help=("If passed as argument, the webcam image will show the "
                          "detected faces. Note that this automatically ensures "
                          "that the camera will be displayed."))
parser.add_argument("-frequency", type=float, default=TIME_BETWEEN_FACE_CHECKS,
                    help=("How often should the camera be queried for a face"))
parser.add_argument('netFile', help="file where the serialized network should be saved")


args = parser.parse_args()

# Parse the user given arguments
displayCam = args.displayWebcam
frequency = args.frequency
displayFaces = args.seeFaces


# When user presses Control-C, gracefully exit program
def signal_handler(signal, frame):
  print "The emotion recognition program will terminate."
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def getCameraCapture():
  with ignoreoutput.suppress_stdout_stderr():
    # 0 is supposed to detected any webcam connected to the device
    capture = cv2.VideoCapture(0)
    if not capture:
      print "Failed VideoCapture: unable to open device 0"
      sys.exit(1)
    return capture

def showFrame(frame, faceCoordinates, emotion=None, draw=False):
  if draw and faceCoordinates:
    #  Draw emotions here as well
    faceRecognition.drawFace(frame, faceCoordinates, emotion)

  cv2.imshow(WINDOW_NAME, frame)

# Currently does not destroy window due to OpenCV issues
def destroyWindow():
  cv2.destroyWindow(WINDOW_NAME)
  cv2.waitKey(1)

def readNetwork():
  with open(args.netFile, "rb") as f:
    net = pickle.load(f)
  return net

def recogintionWork(image, faceCoordinates, net):
  return emotionrecognition.testImage(image, faceCoordinates, net)

# Draw faces argument is only taken into account if display was set as true.
def detectedAndDisplayFaces(capture, net, display=False, drawFaces=False):
  recognition = True
  # Flag gives us some information about the capture
  # Frame is the webcam frame (a numpy image)
  flag, frame = capture.read()
  # Not sure if there is an error from the cam if we should lock the screen
  if flag:
    faceCoordinates = faceRecognition.getFaceCoordinates(frame)
    if faceCoordinates and recognition:
      emotion = recogintionWork(frame, faceCoordinates, net)
    else:
      emotion = None
    if display:
      showFrame(frame, faceCoordinates, emotion, drawFaces)
    if faceCoordinates:
      return True
  else:
    return True

def detectEmotions(frequency, net, display=False, drawFaces=False):
  capture = getCameraCapture()
  if display:
    cv2.startWindowThread()
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)


  while True:
    detectedAndDisplayFaces(capture, net, display, drawFaces)
    time.sleep(frequency)

def main():
  global frequency

  if displayFaces:
    showCam = True
  else:
    showCam = displayCam

  net = readNetwork()
  detectEmotions(frequency, net, showCam, displayFaces)
  destroyWindow()


if __name__ == '__main__':
  main()
