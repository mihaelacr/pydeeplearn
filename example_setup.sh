# This is an example setup of how to ensure all dependencies of pydeeplearn are
# in place. Before running a pydeeplearn program you have might have to ensure
# that theano knows about your GPU and if you linked numpy/ theano with openblas
# where openblas is in your system.


# Enable the virtual env (not required)
source /data/mcr10/theano7/bin/activate
# Setup the cuda paths (I suggest using a script for this, it can get messy fast)
source /vol/cuda/6.5.14/setup.sh

# Tell theano to use the GPU and enable fast math
export THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True'

# Set up the ld library path: required if you use open blas
export LD_LIBRARY_PATH=/data/mcr10/opt/lib:$LD_LIBRARY_PATH

# Set up the library path: required if you use open blas
export LIBRARY_PATH=/data/mcr10/opt/lib:$LIBRARY_PATH
