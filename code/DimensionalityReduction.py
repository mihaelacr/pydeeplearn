# File that implements
# 1.PCA
# 2. Whitened PCA
# 3. LDA
# 4. Locality maps

import numpy as np
import scipy

import itertools
import heapq

def dot(*args):
  return reduce(np.dot, args)

"""
The given matrix has to be symmetric, we are running eigh on it. (a routine
  specially made for Hermitian matrices)
"""
def eigAnalysis(mat):
  eigVals, eigVecs = scipy.linalg.eigh(mat)

  # Get the positive ones
  eigVals = eigVals[eigVals > 0.0]
  eigVecs = eigVecs[:, eigVals > 0.0]
  # Sort in decreasing order
  indices = np.argsort(eigVals)[::-1]
  return eigVals[indices], eigVecs[:, indices]

"""
Arguments:
  data: a numpy array
  dimension: the dimension to reduce the data to (an integer)
  whitened: a boolean. If true, whitened PCA is performed
"""
def PCA(data, dimension=None, whitened=False, getEigenValues=False):
  # The data = the columns, subtract the mean of the colulambdaMmns
  meanData = data.mean(axis=1)
  zeroMean = data - meanData[:, np.newaxis]

  # Transpose it so that it works out like in the lectures
  # This is when the data contains the instances as columns not rows
  nrDataPoints = zeroMean.shape[1]
  F = zeroMean.shape[0]

  # If the user did not provide any dimension, set the dimension
  # to be the min betwen the nr of data points and F
  if dimension == None:
    dimension = min(F, nrDataPoints)

  # now we need to check what dimension to use
  # Case 1: more data points than dimensions
  if F > nrDataPoints:
    if dimension > nrDataPoints:
      print "the dimension you have chosen is too high"
      print "we will reduce the dimension to a better one"
      # Reduce the dimension to this for now
      dimension = nrDataPoints - 1

    S_t = np.dot(zeroMean.T , zeroMean)
    assert S_t.shape == (nrDataPoints, nrDataPoints)
    # Get the eigen values and vectors
    eigVals, eigVecs = eigAnalysis(S_t)

    if dimension > len(eigVals):
      dimension = len(eigVals)
      print "the new dimension is " + str(dimension)

    eigVals = eigVals[0:dimension]
    # IMPORTANT: the columns are the eigen vectors, not the rows
    eigVecs = eigVecs[:, 0: dimension]

    lambdaM = np.diag(1.0 / np.sqrt(eigVals))
    principalComponents = dot(zeroMean, eigVecs, lambdaM)

  # Case 2: more data points than dimensions
  else:
    S_t = np.dot(zeroMean, zeroMean.T)
    # Get the eigen values and vectors
    eigVals, eigVecs = eigAnalysis(S_t)

    if dimension > len(eigVals):
      dimension = len(eigVals)
      print "the new dimension is " + str(dimension)

    eigVals = eigVals[0: dimension]
    # IMPORTANT: the columns are the eigen vectors, not the rows
    eigVecs = eigVecs[:, 0: dimension]

    principalComponents = eigVecs

  if whitened:
    lambdaM = np.diag(1.0/ np.sqrt(eigVals))
    principalComponents = np.dot(principalComponents, lambdaM)

  # Do not forget to return meanData (you should use it for the test cases)
  # Depends on how we do it
  # assert principalComponents.shape == (F, dimension), (" shape should be " +
  #                  str((F, dimension))+ " but is " + str(principalComponents.shape))


  assert eigVals.shape[0] == principalComponents.shape[1]

  if getEigenValues:
    return eigVals, principalComponents
  else:
    return principalComponents


#####################LDA#####################3

# First implementation, using the invert ability
# The data is the columns
def LDAWithoutSimDiagonalization(data, classes, dimension=None):
  # Step1: build the within and between class scatter matrix
  # Let's amke the classes start form 0 so that
  # you can easily make the loops without having to worry about an extra 1
  classes = classes - 1
  distinctClasses = np.unique(classes)
  nrClasses = len(distinctClasses)

  # Get the mean of the data
  meanData = np.mean(data, axis=1)
  # for each class let's compute the within class variance
  # and add them all up to compute the within class scatter matrix
  sw = np.zeros(shape=(data.shape[0], data.shape[0]))
  for cls in xrange(nrClasses):
    dataInstances = data[:, classes==cls]
    sw += np.cov(dataInstances)

  # Compute the between class scatter matrix
  sb = np.zeros(shape=(data.shape[0], data.shape[0]))
  for cls in distinctClasses:
    dataInstances = data[:, classes==cls]
    meanClass = np.mean(dataInstances, axis=1)
    diff = meanClass - meanData
    sb += np.dot(diff, diff.T)

  eigVals, eigVecs = eigAnalysis(np.dot(scipy.linalg.inv(sw), sb))

  # If the user specified a dimension less than the number of non zero eigen values
  # then take that
  if dimension and dimension < eigVecs.shape[1]:
    eigVecs = eigVecs[:, 0: dimension]

  return eigVecs

def LDA(data, classes, dimension):
  # Step1: build the within and between class scatter matrix
  # Let's amke the classes start form 0 so that
  # you can easily make the loops without having to worry about an extra 1
  classes = classes - 1
  distinctClasses = np.unique(classes)
  nrClasses = len(distinctClasses)
  nrData = data.shape[1]
  F = data.shape[0]

  # Get the mean of the data
  meanData = np.mean(data, axis=1)

  # get zero mean data
  zeroMean = data - meanData[:, np.newaxis]

  # Step 2: build the M matrix
  es = []
  for i in xrange(nrClasses):
    # Build E_i for the class
    # N_c_i is the number of instances in this class
    N_c_i = np.sum(classes==i)
    E_i = 1.0 / N_c_i  * np.outer(np.ones(N_c_i) , np.ones(N_c_i))
    es.append(E_i)

  M = scipy.linalg.block_diag(*es)

  assert M.shape == (nrData, nrData)
  # check that M is idempotent
  assert np.allclose(np.dot(M, M), M)

  X_w = np.dot(zeroMean, np.identity(nrData)- M)
  assert X_w.shape == (F, nrData)

  # Do whitened PCA because then you get directly the U matrix
  # IS THERE A PROBLEM THAT WHEN I DO PCA I SUBTRACT THE MEAN?
  eigVecs = PCA(X_w, whitened=True)

  U = eigVecs

  # Compute the projections of the class means using U
  X_b = reduce(np.dot, [U.T, zeroMean, M])

  assert X_b.shape[1] == nrData

  # Compute Q by doing eig analysis on X_b * X_b ^ T
  # No need for whitening here
  eigVecsQ = PCA(X_b, whitened=False)
  assert eigVecsQ.shape[0] == X_b.shape[0]

  # Compute the total transform
  W = np.dot(U, eigVecsQ)

  return W

##########LLP##########################

def buildneightMatrixComplicated(data, neigh):
  nrDataPoints = data.shape[1]

  dataPointToNeightbours  = {}
  dataPointToDistances = {}
  for i in xrange(nrDataPoints):
    dataPointToNeightbours[i] = []
    dataPointToDistances[i] = []
    # With the current algo, the loop has to start at 0 as well (without the row itself)

    for j in xrange(nrDataPoints):
      if i == j:
        continue
      currentNorm = scipy.linalg.norm(data[:, i] - data[:, j])
      if len(dataPointToNeightbours[i]) < neigh:
        dataPointToNeightbours[i].append(j)
        dataPointToDistances[i].append(currentNorm)
      else:
        assert neigh == len(dataPointToNeightbours[i])
        assert neigh == len(dataPointToDistances[i])
        # Replace the biggest norm by the current norm
        # and the index as well
        maxIndex, maxDistance = max(enumerate(dataPointToDistances[i]),
                                   key= lambda x: x[1])
        assert maxIndex <= nrDataPoints
        # We found a point closer than one of the neighbours
        # Swap them
        if currentNorm < maxDistance:
          dataPointToNeightbours[i][maxIndex] = j
          dataPointToDistances[i][maxIndex] = currentNorm

  # Build the S matrix
  S = np.zeros((nrDataPoints, nrDataPoints))
  for i in xrange(nrDataPoints):
    S[i,  dataPointToNeightbours[i]] =  1

  # Ensure that S is symmetric
  S =  np.remainder(S + S.T, 2)
  assert np.allclose(S, S.T)

  return S


def buildneigMatrix(data, neigh):
  nrDataPoints = data.shape[1]

  # Second implementation: does the same thing but in a more succinct way
  dT = data.T
  distancesAndNeighbours = [
    heapq.nsmallest(neigh + 1,
                    [(scipy.linalg.norm(r1 - r2), i) for i, r2 in enumerate(dT)],
                    key=lambda x: x[0])[1:]
    for r1 in dT]
  neighbours = [[x for _, x in val] for val in distancesAndNeighbours]
  distances = [[x for x, _ in val] for val in distancesAndNeighbours]

  # Build the S matrix
  S = np.zeros((nrDataPoints, nrDataPoints))
  for i in xrange(nrDataPoints):
    neighDistances = np.array(distances[i])
    # You can also play with the variance t here
    # S[i,  neighbours[i]] = np.exp(- neighDistances)
    S[i,  neighbours[i]] = 1
    S[neighbours[i], i] = 1

  print S.sum()
  assert np.allclose(S, S.T)
  return S

# This directly finds out the latent variables y
# For a
def LPPDirectLatent(data, dimension=None, neig=20):
  # First step: build the matrix S from the data
  S = buildneigMatrix(data, neig)

  # Second step: get the matrix D from the matrix S
  sums = np.sum(S, axis=0)
  D = np.diag(sums)

  # Let's do the eigen analysis on D^-1 * (D - S)
  M = np.dot(np.linalg.inv(D), D - S)
  eigVals, eigVecs = eigAnalysis(M)

  # The eigenVecs are in the columns, get the first dimension of them
  if dimension is not None:
    return  eigVecs[:, 0:dimension]

  return eigVecs


# Does LPP and returns the principal components to be able to use them for testing
# the data
def LPP(data, dimension=None, neig=20):
  # First step: build the matrix S from the data
  S = buildneightMatrixComplicated(data, neig)

  # Second step: get the matrix D from the matrix S
  sums = np.sum(S, axis=0)
  D = np.diag(sums)

  # Simultaneous diagonalization again, we use our PCA routine
  # W = U Q
  # U ^ T (X D X ^T) U = I
  # so we do PCA on X * D 1/2
  # Firstly create the D^ 1/2 matrix
  DHalf = np.diag(1.0/ np.sqrt(np.diag(D)))

  U = PCA(np.dot(data, DHalf), whitened=True)

  # Now we need to do eigen analysis on U^ T X (D - S) X ^ T U
  _, Q = eigAnalysis(reduce(np.dot, [U.T, data, D - S, data.T, U]))

  eigVecs = np.dot(U, Q)

  if dimension is not None:
    return eigVecs[0, 0: dimension]

  return eigVecs

##################################ICA#########################3

# In the notes we assume whitened data so we might have to prepreocess it
# Preprocessing:
# 1. Centering
# 2. Sphering
# you also need to choose a function G (more than quadratic)

# returns the whitened data
def sphereData(data):
  S_t = np.dot(data, data.T)
  eigVals, eigVecs = eigAnalysis(S_t)
  lambdaM = np.diag(1.0/ np.sqrt(eigVals))
  # Return the whitened reconstructions
  return reduce(np.dot, [eigVecs, lambdaM, eigVecs.T, data])


# Create some G and g functions
# They need to be vectorized, but numpy handles this nicely :-)
def G1(y):
  return 1.0 / 4 * y ** 4

def g1(y):
  return y ** 3

def G2(y, c1=1.0):
  return - 1.0 / c1 * np.exp(- c1 / 2 * y**2)

def g2(y, c1=1.0):
  return y * np.exp(- c1 / 2 * y**2)

def G3(y, c2=1.0):
  return 1.0 / c2 * np.log(np.cosh(c2 * y))

def g3(y, c2=1.0):
  return np.tanh(c2 * y)


# Implementation of ICA
# G and g have to be vectorized functions
def ICA(data, G, g, random=True):
  # Step1: preprocessing, centering
  meanData = data.mean(axis=1)
  zeroMean = data - meanData[:, np.newaxis]

  # Step 2: preprocessing, centering
  data = sphereData(zeroMean)

  F = data.shape[0]
  nrData = data.shape[1]

  X = data

  # Step 3: Apply newton updates to find out the columns of A = W ^ -1
  # What is the size of the independent components?
  # It should be the same size as the data as W is a square matrix

  elemsOfA = []

  for i in xrange(nrData):
    # Find the first component
    if random:
      a = np.random.random_sample(F)
    else:
      a = np.ones(F)
    # Until convergence: TODO: change this, make sure that
    # you will stop when you should do
    # Make the update: (would this not work better if you have a learning rate?)
    for i in xrange(100):
      # Write the means nicely so that you get them as a matrix multiplication thing
      # This is to get the expectation and the update

      # First compute E(g(a^T*x_i)), and we compute this by using the sample
      # average E(g(a^T*x_i)) = 1/N * sum_i g(a^T*x_i)
      # In order to take advantage of fast matrix operations, do it
      # using matrix multiplication
      a_X = np.dot(a, X)
      exp1 = np.mean(g(a_X))
      # Exp1 needs to be a number
      assert len(exp1.shape) == 0

      # Now compute E(x_i*G(a^T*x_i)), in a similar fashion, by using
      # matrix multiplication
      # exp2 needs to be a vector
      exp2 = np.dot(G(a_X), X.T) / nrData

      a = exp2 - exp1 * a

      # Now we have to apply the decorrelation step
      for previousA in elemsOfA:
        a -= np.dot(a, previousA) * previousA

      # Normalization
      a /= np.linalg.norm(a)

    # We found a, remember it
    elemsOfA.append(a)

  A = np.array(elemsOfA)

  # return A.T, as we transpose again in later code
  return A.T
