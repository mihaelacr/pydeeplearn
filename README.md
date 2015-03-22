pydeeplearn
===========

Library for deep belief nets and rbms  and simple CNNs. Provides code for dropout, rmsprop rectified linear units, sparsity constraints, weight decay, etc. I recently added support for [adversarial training](https://drive.google.com/file/d/0B64011x02sIkX0poOGVyZDI4dUU/view). Runs on GPUs for high performance by using [theano](http://deeplearning.net/software/theano/).


For more details see the [report](http://elarosca.net/report.pdf) of the project which used this code and the [ poster](http://elarosca.net/poster.pdf) submitted for the participation to the SET awards. The project was made as part of the requirements for a master degree at Imperial College London and received a [prize of excellence] (http://www3.imperial.ac.uk/computing/teaching/ug/ug-distinguished-projects).

One of the key points of this implementation and API is that they do not impose theano on the user. While theano is a great tool it has quite a steep learning curve so I decided to take advantage of its power without imposing that the user learns theano. Hence all the interface code just requires python and basic numpy knowledge. To the best of my knowledge this is the only library with these features.

The API provided by DBNs is compatible with scikit learn so you can use all the functionality from that library in conjuction with my implementation.

 In case you use my code for a paper, study or project please cite my report and if you can, add a link to this repository. 

# User guide
  * The library is in `code/lib/`. You can find there the implementations of RBMs, CNNs and DBNs.
  * Multiple examples on how to use RBMs, DBNs and CNNs are in `code/MNISTdigits.py` and `code/emotions.py`
  * The code that implements a network which determines the similarity between two inputs is in `code/similarity` 
  * The old code that is not based on theano but only numpy is in `code/old-version`. This code is incomplete. Do not use it. It is there for educational purposes because it is easier to understand how to implement RBMs and DBNs without theano.
  * If you are a beginner in deep learning, please check out my [report](http://elarosca.net/report.pdf). It explains the foundation behind the concepts used in this library.
  * If you still have questions, pop me an email or a message.

# Future and current work
 I am currently working on completing the CNN implementation and thinking of adding Bayesian optimization to avoid doing CV for finding out hyper parameters. When [whetlab](https://www.whetlab.com) will be in open beta, I will integrate with their UI as well. Currently, see the [spearmint](https://github.com/mihaelacr/pydeeplearn/tree/spearmint) branch for an example of how to use spearmint with pydeeplearn. This is still experimental, I am still playing around with it so it's not yet into master.
 
 I have added support for [adversarial training](https://drive.google.com/file/d/0B64011x02sIkX0poOGVyZDI4dUU/view) but it is not merged in master yet. If you are interested, check out the [adversarialexamples branch](https://github.com/mihaelacr/pydeeplearn/tree/adversarialexamples).
  
 If you have any feature requests, please let me know.
 
# How to install dependencies 


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

Note that for theano to work on the GPU you need cuda installed and setting up some environment variables.

`PATH` needs to containt the path to nvcc (usually `/usr/local/cuda/bin`)

`CUDA_PATH` needs to contain the path to cuda (usually `/usr/local/cuda/`)

`LD_LIBRARY_PATH` needs to contain the linker libraries for cuda (`/usr/local/cuda/lib64`)

`LD_LIBRARY_PATH` needs to contain the path to openblas (for numpy to find it: this is the prefix path you chose at step 3) and the path to OpenCV in case it was not installed globally

`LIBRARY_PATH` needs to contain the path to the nvidia driver (something like `/usr/lib/nvidia-331`)

# Acknowledgements

I would like to thank the Department of Computing at Imperial College and Prof. Abbas Edalat for their support during my thesis and for allowing me to continue with my experiments on lab equipment after graduation.
