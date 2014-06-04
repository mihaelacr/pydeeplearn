#!/usr/bin/env bash


# Also activate the virtual env that has theano and all that
# source /data/mcr10/myenv/bin/activate

export PATH=/usr/local/cuda/bin:$PATH
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/nvidia-331:$LIBRARY_PATH
export LD_LIBRARY_PATH=/data/mcr10/opt/lib:$LD_LIBRARY_PATH


export CUDA_ROOT=/usr/local/export

# for opencv
export LD_LIBRARY_PATH=/data/mcr10/opencv/lib/:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/data/mcr10/opencv/lib/pkgconfig:$PKG_CONFIG_PATH
export PYTHONPATH=/data/mcr10/opencv/lib/python2.7/site-packages
PYTHONPATH=$PYTHONPATH:/path/to/cudamat

# For cudamat: trial in nolearn
export PYTHONPATH=$PYTHONPATH:/data/mcr10/theanoopenblas/cudamat

export THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True'
