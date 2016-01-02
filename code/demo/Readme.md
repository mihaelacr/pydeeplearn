# Emotion recognition from webcam capture

This directory contains the code that does emotion recognition from the camera.

This code was used to obtain the demo [here](http://elarosca.net/video.ogv).

## Strategy
The code uses OpenCV to detect faces from the webcam stream and then uses an emotion classifier to detect the emotion.

## Preprocessing
The webcam detected images are processed as follows:
  * Cropped according to the rectangle suggested by OpenCV as the face
  * Takes the image into black and white
  * Uses histogram normalization
  * Data is scaled to have zero mean and unit variance
  
The same preprocessing should be used on the input data when training the emotion classifier.

## Emotion classifier
 The emotion classifier I used for the [demo](http://elarosca.net/video.ogv) was a DBN network trained usign `pydeeplearn`.
 
 There is no need for the user of the code under this directory to use `pydeeplearn` as the emotion classifier. The user can pass in as a flag a pickle file of a model that has a `classify` method:
 
 ```model.classify(image) ```
 that returns the probabilities obtained from the network as well as the classification label.
 
 Replacing a `pydeeplearn` classifier with another classifier can be made easier. If you are interested in that, please either send a pull request or create an issue. 
 
## Recognition for icons
  * Happy emotiocon: Person by Catherine Please from The Noun Project
  * Sad emoticon: Sad by Cengiz SARI from The Noun Project
  * Surprised emoticon: Surprise designed by Chris McDonnell from the thenounproject.com
