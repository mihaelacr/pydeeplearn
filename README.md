pydeeplearn
===========

Library for deep belief nets and rbms and convolutional neural networks (simple version). Provides code for dropout, rmsprop, momentum, rectified linear units, sparsity constraints, weight decay, adversarial training, etc. Runs on GPUs for high performance by using [theano](http://deeplearning.net/software/theano/).


For more details see the [report](http://elarosca.net/report.pdf) of the project which used this code and the [ poster](http://elarosca.net/poster.pdf) submitted for the participation to the SET awards. The project was made as part of the requirements for a master degree at Imperial College London and received a [prize of excellence](http://www3.imperial.ac.uk/computing/teaching/ug/ug-distinguished-projects).

One of the key points of this implementation and API is that they do not impose theano on the user. While theano is a great tool it has quite a steep learning curve so I decided to take advantage of its power without imposing that the user learns theano. Hence all the interface code just requires python and basic numpy knowledge. To the best of my knowledge this is the only library with these features.

The API provided by DBNs is compatible with scikit learn so you can use all the functionality from that library in conjuction with my implementation.

 In case you use my code for a paper, study or project please cite my report and if you can, add a link to this repository.

## Demo video
I used pydeeplearn and openCV to make an application which detects emotions live from the webcam stream. You can see a demo video of me fooling around at the camera here: http://elarosca.net/video.ogv

## User guide
  * The library is in `code/lib/`. You can find there the implementations of RBMs, CNNs and DBNs.
  * Multiple examples on how to use RBMs, DBNs and CNNs are in `code/MNISTdigits.py` and `code/emotions.py`
  * The code that implements a network which determines the similarity between two inputs is in `code/similarity`
  * The old code that is not based on theano but only numpy is in `code/old-version`. This code is incomplete. Do not use it. It is there for educational purposes because it is easier to understand how to implement RBMs and DBNs without theano.
  * If you are a beginner in deep learning, please check out my [report](http://elarosca.net/report.pdf). It explains the foundation behind the concepts used in this library.
  * If you still have questions, pop me an email or a message.

## NEW: Docker container
If you do not want to go trough all the hurdle of installing the dependencies needed for pydeeplearn then you can just use the docker container found at on [docker hub](https://hub.docker.com/r/mihaelacr/pydeeplearn-labeled/).
The docker container comes with the MNIST digits so you do not have to download the files yourself.

For GPU usage, I suggest using the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) wrapper to ensure that docker works with the GPU.

These instructions should get you up and running with `pydeeplearn`:
  ```
  git clone https://github.com/NVIDIA/nvidia-docker.git
  cd nvidia-docker
  GPU=0 ./nvidia-docker run --rm -it mihaelacr/pydeeplearn-labeled bash
  cd pydeeplearn/code
  ```

Check that pydeeplearn works:
 ```
 THEANO_FLAGS='device=gpu,floatX=float32' PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python MNISTdigits.py --trainSize 1000 --testSize 10 --db --train --rbmnesterov test.p --save
 ```
 
 If you just want to check that theano works with the GPU then just try:
 ```
 THEANO_FLAGS='device=gpu,floatX=float32' PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64 python -c 'import theano'
 ```
 This should print something like `Using GPU device...`.

## Key features

### Network types
  * RBM
  * DBN
  * CNN
  * ANN (trough the dbn.py implementation by setting preTrainEpochs=0 in the constructor)
  * similarity networks (with RBM siamese networks)


### Training tricks supported
  * early stopping
  * simple momentum
  * Nesterov momentum
  * dropout (for the hidden and visible layer)
  * adversarial training 
  * rmsprop
  * scaling the learning rate by momentum
  * multiple activation functions (and with ease we can support more)
  * integration with bayesian optimization framework for hyperparamter optimization (spearmint)
  * multiple hidden unit types (binary, real valued)

Supported image preprocessing:
  * histogram equalization (using openCV)
  * face cropping (for face images, using openCV) 

## Future and current work
For the future plans that I have for the library please see the TODO.txt file. Note that currently pydeeplearn is a side project for me and some features might take some time to implement. 

If you want a feature implemented, please either send a pull request or let me know. I will do my best to get it up and running for you.

## Running examples

### MNIST
In order to be able to use the MNIST digits examples, you have to first get the data from the [official website](http://yann.lecun.com/exdb/mnist/). The code/MNISTdigits.py script reads the data using the --path argument, so you must set that argument to point to the directory in which you now have the data. In order to see the different options available for training and testing a network for digit recognition, see the possible flags in MNISTdigits.py. Note that an unnamed containing a file for the stored network is required as a final argument: if training is performed (decided with the --train flag), the resulting network will be stored in that file, and if no training is performed, a network is assumed to be already stored in that file and will be retrivied using pickle. 

Example run:

  `python MNISTdigits.py --trainSize 60000 --testSize 10000 --nesterov --rbmnesterov --maxEpochs 1000  --miniBatchSize 20  --rmsprop  network.p `

### Emotion recognition
The script in code/emotions.py contains code on how to do emotion recognition from images using deep belief networks. The code there uses multiple datasets: Jaffe, Chon-Kanade, MultiPie and other other unlabelled datasets. While Jaffe and Chon-Kanade are publically available, the MultiPie dataset is available only via purchase. Some code also hadndles the data available in a previous Kaggle competition (data can be found [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data). Note that I did not spend much time optimizing hyperparameters for the Kaggle compeition data, so if you are interested in obtaining better results you might start from using spearmint to obtain good values for the hyperparameters.  

### Similarity detection using siamese networks
 You can train a network to detect if two images contain represent the same person or the same emotion using code in `code/similarity`. Note that in order to be able to train such a network, labelled data (with subjects or emotions) is needed. Most of the labelled data that I used for these experiments was taken from the MultiPie dataset. The code can be run as follows:
 
 `python similarityMain.py --diffsubjects --relu  --epochs 90 --rbmepochs 10`

## Cloning the repo
By now `pydeeplearn` has a big git history. If you do not want to get all of it (and you probably do not need it) use:

  `git clone https://github.com/mihaelacr/pydeeplearn.git --depth 1`

## How to install dependencies

1. Create a python virtualenv

2. Clone numpy

 `git clone https://github.com/numpy/numpy`

3. Optional: setup numpy to work with OpenBlas
  `git clone git://github.com/xianyi/OpenBLAS`
  `cd OpenBLAS && make FC=gfortran`
  `sudo make PREFIX=prefixpath install`
  `ldconfig (requires sudo if prefixpath is a global path)`

  in the directory numpy was downloaded:
  `vim site.cfg`
  set the contents to:

  `[atlas]`
  `atlas_libs = openblas`
  `library_dirs = prefixpath/lib`

  or for numpy 19.dev

 ` [openblas]`
  `libraries = openblas`
  `library_dirs = prefixpath/lib`
  `include_dirs = prefixpath/include`

4. `python setup.py config` to ensure everything is OK
5. `python setup.py build && python setup.py install`
6. `pip install --upgrade scikit-learn`
7. `pip install --upgrade cython`
8. `pip install --upgrade scikit-image`
9. `pip install theano`
10. install opencv for python (try the latest version) [for opencv3.0.0-dev](http://docs.opencv.org/trunk/doc/tutorials/introduction/linux_install/linux_install.html)

11. install matplotlib
   `easy_install -m matplotlib`
12. install sklearn

   See the instructions [here](http://scikit-learn.org/stable/install.html)

## Set up

When running a pydeeplearn program you might have to set up some environment variables, depending on your configuration. If want to use the GPU for training/testing a model, you have to ensure that theano knows where your CUDA installation is (for detailed instructions see below).

### Setting up theano to work on the GPU

  `PATH` needs to contain the path to nvcc (usually `/usr/local/cuda/bin`)

  `CUDA_PATH` needs to contain the path to cuda (usually `/usr/local/cuda/`)

  `LD_LIBRARY_PATH` needs to contain the linker libraries for cuda (`/usr/local/cuda/lib64`)

  `LIBRARY_PATH` needs to contain the path to the nvidia driver (something like `/usr/lib/nvidia-331`)


  If you are not configuring theano globally (in /home/user/.theano), then you have to set up the THEANO_FLAGS variable:

  `export THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True'`

### Setting up numpy/ theano to work with openblas

  `LD_LIBRARY_PATH` needs to contain the path to openblas (for numpy to find it: this is the prefix path you chose at step 3 in the installation instructions above) and the path to OpenCV in case it was not installed globally

  `LIBRARY_PATH` needs to contain the path to openblas (for numpy to find it: this is the prefix path you chose at step 3 in the installation instructions above) and the path to OpenCV in case it was not installed globally


## Acknowledgements

I would like to thank the Department of Computing at Imperial College and Prof. Abbas Edalat for their supoprt during my thesis and for allowing me to continue with my experiments on lab equipment after graduation.
