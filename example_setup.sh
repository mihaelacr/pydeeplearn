# Enable the virtual env
source /data/mcr10/new/theanoopenblasnew/bin/activate
# Setup the cude data
source /vol/cuda/6.5.14/setup.sh

export THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True'

# Set up the ld library path
export LD_LIBRARY_PATH=/data/mcr10/opt/lib:$LD_LIBRARY_PATH
# Set up the library path
export LIBRARY_PATH=/data/mcr10/opt/lib:$LIBRARY_PATH
