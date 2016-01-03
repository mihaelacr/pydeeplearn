# Emotion recognition from webcam capture

This directory contains the code that does emotion recognition from the camera.

This code was used to obtain the demo [here](http://elarosca.net/video.ogv).

## Applications
Apart from the theoretical applications, you can use this functionality to control your computer. For example, you can record certain specific expressions and associate them with a command that you want the computer to exectute. For example, you can lock your screen when you close both your eyes very strongly and open your favourite browser when you make a big grin. For best results you can train the classifier with pictures of yourself with the code provided in this directory, as explained below.

## Strategy
The code uses OpenCV to detect faces from the webcam stream and then uses an emotion classifier to detect the emotion.

## Preprocessing
The webcam detected images are processed as follows:
  * Cropped according to the rectangle suggested by OpenCV as the face
  * Takes the image into black and white
  * Uses histogram normalization
  * Data is scaled to have zero mean and unit variance
  
The same preprocessing should be used on the input data when training the emotion classifier.

## Example run
``` python emotionweb.py --displayWebcam --seeFaces --netFile mihaela-happy-sad.p ```

Note: `mihaela-happy-sad.p` is a network trained with pictures of me. It works very well for my face, but might not work for others (I wear glasses and I have long hair, using a network trained with the standard databases did not work well as they do not contain pictures of people wearing glasses and also not a lot of women are present).

## Getting training data

To get best results (and tailored for the person who is using the webcam app), you can use the `emotionweb.py` script to record data, as follows: 

```python emotionweb.py --displayWebcam --seeFaces --gather_training_data  --recording_emotion sad```

## Training a `pydeeplearn` model
If you have recorded your data as explained above, you can train a `pydeeplearn` model using the following command: 
```python train-emotion-net.py --emotions happy sad --display_example_data --path_to_data . --net_file trained.p```

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
