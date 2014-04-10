import numpy as np

import restrictedBoltzmannMachine as rbm
import theano
from theano import tensor as T
from theano.ifelse import ifelse as theanoifelse
from theano.tensor.shared_randomstreams import RandomStreams

theanoFloat  = theano.config.floatX

"""In all the above topLayer does not mean the top most layer, but rather the
layer above the current one."""

from common import *
from debug import *

DEBUG = False
# TODO: the random stream can be put in the
# class: maybe that is faster?

class MiniBatchTrainer(object):

  # TODO: maybe creating the ring here might be better?
  def __init__(self, input, nrLayers, initialWeights, initialBiases,
               visibleDropout, hiddenDropout):
    self.input = input

    # Let's initialize the fields
    # The weights and biases, make them shared variables
    self.weights = []
    self.biases = []
    for i in xrange(nrLayers - 1):
      w = theano.shared(value=np.asarray(initialWeights[i],
                                         dtype=theanoFloat),
                        name='W')
      self.weights.append(w)

      b = theano.shared(value=np.asarray(initialBiases[i],
                                         dtype=theanoFloat),
                        name='b')
      self.biases.append(b)

    # Set the parameters of the object
    # Do not set more than this, these will be used for differentiation in the
    # gradient
    self.params = self.weights + self.biases

    # Required for momentum
    # The updates that were performed in the last batch
    # It is important that the order in which
    # we add the oldUpdates is the same as which we add the params
    # TODO: add an assertion for this
    self.oldUpdates = []
    for i in xrange(nrLayers - 1):
      oldDw = theano.shared(value=np.zeros(shape=initialWeights[i].shape,
                                           dtype=theanoFloat),
                        name='oldDw')
      self.oldUpdates.append(oldDw)

    for i in xrange(nrLayers - 1):
      oldDb = theano.shared(value=np.zeros(shape=initialBiases[i].shape,
                                           dtype=theanoFloat),
                        name='oldDb')
      self.oldUpdates.append(oldDb)

    # Rmsprop
    # The old mean that were performed in the last batch
    self.oldMeanSquare = []
    for i in xrange(nrLayers - 1):
      oldDw = theano.shared(value=np.zeros(shape=initialWeights[i].shape,
                                           dtype=theanoFloat),
                        name='oldDw')
      self.oldMeanSquare.append(oldDw)

    for i in xrange(nrLayers - 1):
      oldDb = theano.shared(value=np.zeros(shape=initialBiases[i].shape,
                                           dtype=theanoFloat),
                        name='oldDb')
      self.oldMeanSquare.append(oldDb)


    # Create a theano random number generator
    # Required to sample units for dropout
    # If it is not shared, does it update when we do the
    # when we go to another function call?
    self.theano_rng = RandomStreams(seed=np.random.randint(1, 1000))

    # Sample from the visible layer
    # Get the mask that is used for the visible units
    dropout_mask = self.theano_rng.binomial(n=1, p=visibleDropout,
                                            size=self.input.shape,
                                            dtype=theanoFloat)
    # Optimization: only update the mask when we actually sample
    # dropout_mask.rng.default_update =\
    #         theanoifelse(T.lt(visibleDropout, 1.0),
    #                       dropout_mask.rng.default_update,
    #                       dropout_mask.rng)

    currentLayerValues = self.input * dropout_mask

    for stage in xrange(len(self.weights) -1):
      w = self.weights[stage]
      b = self.biases[stage]
      linearSum = T.dot(currentLayerValues, w) + b
      # TODO: make this a function that you pass around
      # it is important to make the classification activation functions outside
      # Also check the Stamford paper again to what they did to average out
      # the results with softmax and regression layers?
        # Use hiddenDropout: give the next layer only some of the units
        # from this layer
      dropout_mask = self.theano_rng.binomial(n=1, p=hiddenDropout,
                                            size=linearSum.shape,
                                            dtype=theanoFloat)
        # Optimization: only update the mask when we actually sample
        # dropout_mask.rng.default_update =\
        #     theanoifelse(T.lt(hiddenDropout, 1.0),
        #                   dropout_mask.rng.default_update,
        #                   dropout_mask.rng)
      currentLayerValues = dropout_mask * T.nnet.sigmoid(linearSum)

    # Do not use theano's softmax, it is numerically unstable
    # and it causes Nans to appear
    # Note that semantically this
    e_x = T.exp(linearSum - linearSum.max(axis=1, keepdims=True))
    currentLayerValues = e_x / e_x.sum(axis=1, keepdims=True)

    self.output = currentLayerValues

  def cost(self, y):
    return T.nnet.categorical_crossentropy(self.output, y)

""" Class that implements a deep belief network, for classification """
class DBN(object):

  """
  Arguments:
    nrLayers: the number of layers of the network. In case of discriminative
        traning, also contains the classifcation layer
        (the last softmax layer)
        type: integer
    layerSizes: the sizes of the individual layers.
        type: list of integers of size nrLayers
  """
  def __init__(self, nrLayers, layerSizes,
                unsupervisedLearningRate=0.01,
                supervisedLearningRate=0.05,
                nesterovMomentum=True,
                rmsprop=True,
                miniBatchSize=10,
                hiddenDropout=0.5,
                rbmHiddenDropout=0.5,
                visibleDropout=0.8,
                rbmVisibleDropout=1):
    self.nrLayers = nrLayers
    self.layerSizes = layerSizes

    assert len(layerSizes) == nrLayers
    self.hiddenDropout = hiddenDropout
    self.visibleDropout = visibleDropout
    self.rbmHiddenDropout = rbmHiddenDropout
    self.rbmVisibleDropout = rbmVisibleDropout
    self.miniBatchSize = miniBatchSize
    self.supervisedLearningRate = supervisedLearningRate
    self.unsupervisedLearningRate = unsupervisedLearningRate
    self.nesterovMomentum = nesterovMomentum
    self.rmsprop = rmsprop

  """
    Choose a percentage (percentValidation) of the data given to be
    validation data, used for early stopping of the model.
  """
  def train(self, data, labels, percentValidation=0.1):
    nrInstances = len(data)
    validationIndices = np.random.choice(xrange(nrInstances),
                                         percentValidation * nrInstances)
    trainingIndices = list(set(xrange(nrInstances)) - set(validationIndices))
    trainingData = data[trainingIndices, :]
    trainingLabels = labels[trainingIndices, :]

    validationData = data[validationIndices, :]
    validationLabels = labels[validationIndices, :]

    self.trainWithGivenValidationSet(trainingData, trainingLabels,
                                     validationData, validationLabels)

  def trainWithGivenValidationSet(self, data, labels,
                                  validationData, validationLabels):
    nrRbms = self.nrLayers - 2

    self.weights = []
    self.biases = []

    sharedData = theano.shared(np.asarray(data, dtype=theanoFloat))
    sharedLabels = theano.shared(np.asarray(labels, dtype=theanoFloat))

    sharedValidationData = theano.shared(np.asarray(validationData, dtype=theanoFloat))
    sharedValidationLabels = theano.shared(np.asarray(validationLabels, dtype=theanoFloat))

    # Train the restricted Boltzmann machines that form the network
    currentData = data
    for i in xrange(nrRbms):
      net = rbm.RBM(self.layerSizes[i], self.layerSizes[i+1],
                    learningRate=self.unsupervisedLearningRate,
                    hiddenDropout=self.rbmHiddenDropout,
                    visibleDropout=self.rbmVisibleDropout)
      net.train(currentData)

      w = net.testWeights
      self.weights += [w / self.hiddenDropout]
      # Only add the biases for the hidden unit
      b = net.biases[1]
      self.biases += [b]

      # Let's update the current representation given to the next RBM
      currentData = net.hiddenRepresentation(currentData)

    # This depends if you have generative or not
    # Initialize the last layer of weights to zero if you have
    # a discriminative net
    lastLayerWeights = np.zeros(shape=(self.layerSizes[-2], self.layerSizes[-1]),
                                dtype=theanoFloat)
    lastLayerBiases = np.zeros(shape=(self.layerSizes[-1]),
                               dtype=theanoFloat)

    self.weights += [lastLayerWeights]
    self.biases += [lastLayerBiases]

    assert len(self.weights) == self.nrLayers - 1
    assert len(self.biases) == self.nrLayers - 1

    self.nrMiniBatches = len(data) / self.miniBatchSize

    # Does backprop for the data and a the end sets the weights
    self.fineTune(sharedData, sharedLabels,
                  sharedValidationData, sharedValidationLabels)

    # Get the classification weights
    self.classifcationWeights = map(lambda x: x * self.hiddenDropout, self.weights)
    self.classifcationBiases = self.biases

  """Fine tunes the weigths and biases using backpropagation.
    data and labels are shared

    Arguments:
      data: The data used for traning and fine tuning
        data has to be a theano variable for it to work in the current version
      labels: A numpy nd array. Each label should be transformed into a binary
          base vector before passed into this function.
      miniBatch: The number of instances to be used in a miniBatch
      epochs: The number of epochs to use for fine tuning
  """
  def fineTune(self, data, labels, validationData, validationLabels,
               maxEpochs=200):
    print "supervisedLearningRate"
    print self.supervisedLearningRate
    batchLearningRate = self.supervisedLearningRate / self.miniBatchSize
    batchLearningRate = np.float32(batchLearningRate)

    nrMiniBatches = self.nrMiniBatches
    # Let's build the symbolic graph which takes the data trough the network
    # allocate symbolic variables for the data
    # index of a mini-batch
    miniBatchIndex = T.lscalar()
    momentum = T.fscalar()

    # The mini-batch data is a matrix
    x = T.matrix('x', dtype=theanoFloat)
    # labels[start:end] this needs to be a matrix because we output probabilities
    y = T.matrix('y', dtype=theanoFloat)

    batchTrainer = MiniBatchTrainer(input=x, nrLayers=self.nrLayers,
                                    initialWeights=self.weights,
                                    initialBiases=self.biases,
                                    visibleDropout=0.8,
                                    hiddenDropout=0.5)

    # the error is the sum of the errors in the individual cases
    error = T.sum(batchTrainer.cost(y))
    deltaParams = T.grad(error, batchTrainer.params)

    if DEBUG:
      mode = theano.compile.MonitorMode(post_func=detect_nan).excluding(
                                        'local_elemwise_fusion', 'inplace')
    else:
      mode = None

    if self.nesterovMomentum:
      preDeltaUpdates, updates = self.buildUpdatesNesterov(batchTrainer, momentum,
                    batchLearningRate, deltaParams)
      momentum_step = theano.function(
          inputs=[momentum],
          outputs=[],
          updates=preDeltaUpdates,
          mode = mode)

      update_params = theano.function(
          inputs =[miniBatchIndex],
          outputs=error,
          updates=updates,
          givens={
              x: data[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize],
              y: labels[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize]},
          mode=mode)

      def trainModel(miniBatchIndex, momentum):
        momentum_step(momentum)
        return update_params(miniBatchIndex)
    else:

      updates = self.buildUpdatesSimpleMomentum(batchTrainer, momentum,
                    batchLearningRate, deltaParams)
      trainModel = theano.function(
            inputs=[miniBatchIndex, momentum],
            outputs=error,
            updates=updates,
            givens={
                x: data[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize],
                y: labels[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize]})

    # Let's create the function that validates the model!
    validate_model = theano.function(inputs=[],
      outputs=batchTrainer.cost(y),
      givens={x: validationData,
              y: validationLabels})

    lastValidationError = np.inf
    count = 0
    epoch = 0

    # while epoch < maxEpochs and count < 5:
    for epoch in xrange(100):
      print "epoch"
      if epoch < maxEpochs / 10:
        momentum = np.float32(0.5)
      else:
        momentum = np.float32(0.95)

      for batchNr in xrange(nrMiniBatches):
        trainModel(batchNr, momentum)

      meanValidation = np.mean(validate_model(), axis=0)

      if meanValidation > lastValidationError:
          count +=1
      else:
          count = 0
      lastValidationError = meanValidation

      epoch +=1

    # Set up the weights in the dbn object
    for i in xrange(len(self.weights)):
      self.weights[i] = batchTrainer.weights[i].get_value()

    for i in xrange(len(self.biases)):
      self.biases[i] = batchTrainer.biases[i].get_value()

    print "number of epochs"
    print epoch


  def buildUpdatesNesterov(self, batchTrainer, momentum,
                  batchLearningRate, deltaParams):
    preDeltaUpdates = []
    for param, oldUpdate in zip(batchTrainer.params, batchTrainer.oldUpdates):
      newParam = param + momentum * oldUpdate
      preDeltaUpdates.append((param, newParam))

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    parametersTuples = zip(batchTrainer.params,
                           deltaParams,
                           batchTrainer.oldUpdates,
                           batchTrainer.oldMeanSquare)

    for param, delta, oldUpdate, oldMeanSquare in parametersTuples:
      if self.rmsprop:
        meanSquare = 0.9 * oldMeanSquare + 0.1 * delta ** 2
        paramUpdate = - batchLearningRate * delta / T.sqrt(meanSquare + 1e-8)
        updates.append((oldMeanSquare, meanSquare))
      else:
        paramUpdate = - batchLearningRate * delta

      newParam = param + paramUpdate
      updates.append((param, newParam))
      updates.append((oldUpdate, paramUpdate))

    return preDeltaUpdates, updates

  def buildUpdatesSimpleMomentum(self, batchTrainer, momentum,
                  batchLearningRate, deltaParams):
    updates = []
    parametersTuples = zip(batchTrainer.params,
                           deltaParams,
                           batchTrainer.oldUpdates,
                           batchTrainer.oldMeanSquare)

    for param, delta, oldUpdate, oldMeanSquare in parametersTuples:
      paramUpdate = momentum * oldUpdate
      if self.rmsprop:
        meanSquare = 0.9 * oldMeanSquare + 0.1 * delta ** 2
        paramUpdate += - batchLearningRate * delta / T.sqrt(meanSquare + 1e-8)
        updates.append((oldMeanSquare, meanSquare))
      else:
        paramUpdate += - batchLearningRate * delta

      newParam = param + paramUpdate
      updates.append((param, newParam))
      updates.append((oldUpdate, paramUpdate))

    return updates


  def classify(self, dataInstaces):
    dataInstacesConverted = np.asarray(dataInstaces, dtype=theanoFloat)

    x = T.matrix('x', dtype=theanoFloat)

    # Use the classification weights because now we have hiddenDropout
    # Ensure that you have no hiddenDropout in classification
    # TODO: are the variables still shared? or can we make a new one?
    batchTrainer = MiniBatchTrainer(input=x, nrLayers=self.nrLayers,
                                    initialWeights=self.classifcationWeights,
                                    initialBiases=self.classifcationBiases,
                                    visibleDropout=1,
                                    hiddenDropout=1)

    classify = theano.function(
            inputs=[],
            outputs=batchTrainer.output,
            updates={},
            givens={x: dataInstacesConverted})

    lastLayers = classify()

    return lastLayers, np.argmax(lastLayers, axis=1)
