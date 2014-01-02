import scipy

"""
Arguments:
  vec: A numpy 1-D vector.
  size: A 2D tuple

Returns:
  A 2-D vector of dimension 'size', only if 'vec' has compatible dimensions.
  Otherwise it throws an error.
"""
def vectorToImage(vec, size):
  return vec.reshape(size)

""" Transforms the 2D images into 1D vectors
Arguments:
  images: is a python list of numpy arrays
Returns:
  A python list of 1-D numpy arrays, transformed from the input 2D ones
  No data is lost in the transformation.
"""
def imagesToVectors(images):
  return scipy.array(map(lambda x: x.reshape(-1), images))


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

sigmoidVec = np.vectorize(sigmoid, otypes=[np.float])

# Takes the value of the sigmoid function and returns the derivative
def sigmoidDerivativeFromValue(sigmoidValue):
  return sigmoidValue * (1 - sigmoidValue)

sigmoidDerivativeVec = np.vectorize(sigmoidDerivativeFromValue, otypes=[np.float])
