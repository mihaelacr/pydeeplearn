import numpy as np

import restrictedBoltzmannMachine as rbm
from activationfunctions import *

import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

theanoFloat  = theano.config.floatX

"""In all the above topLayer does not mean the top most layer, but rather the
layer above the current one."""

from common import *
from debug import *

DEBUG = False

class MiniBatchTrainer(object):

  def __init__(self, input, nrLayers, initialWeights, initialBiases,
               activationFunction, classificationActivationFunction,
               visibleDropout, hiddenDropout):
    self.input = input

    # Let's initialize the fields
    # The weights and biases, make them shared variables
    self.weights = []
    self.biases = []
    nrWeights = nrLayers - 1
    for i in xrange(nrWeights):
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
    # It is important that the order in which we add the oldUpdates is the same
    # as which we add the params
    self.oldUpdates = []
    for i in xrange(nrWeights):
      oldDw = theano.shared(value=np.zeros(shape=initialWeights[i].shape,
                                           dtype=theanoFloat),
                        name='oldDw')
      self.oldUpdates.append(oldDw)

    for i in xrange(nrWeights):
      oldDb = theano.shared(value=np.zeros(shape=initialBiases[i].shape,
                                           dtype=theanoFloat),
                        name='oldDb')
      self.oldUpdates.append(oldDb)

    # Rmsprop
    # The old mean that were performed in the last batch
    self.oldMeanSquare = []
    for i in xrange(nrWeights):
      oldDw = theano.shared(value=np.zeros(shape=initialWeights[i].shape,
                                           dtype=theanoFloat),
                        name='oldDw')
      self.oldMeanSquare.append(oldDw)

    for i in xrange(nrWeights):
      oldDb = theano.shared(value=np.zeros(shape=initialBiases[i].shape,
                                           dtype=theanoFloat),
                        name='oldDb')
      self.oldMeanSquare.append(oldDb)

    # Create a theano random number generator
    # Required to sample units for dropout
    self.theanoRng = RandomStreams(seed=np.random.randint(1, 1000))

    # Sample from the visible layer
    # Get the mask that is used for the visible units
    if visibleDropout in [1.0, 1]:
      currentLayerValues = self.input
    else:
      dropoutMask = self.theanoRng.binomial(n=1, p=visibleDropout,
                                            size=self.input.shape,
                                            dtype=theanoFloat)
      currentLayerValues = self.input * dropoutMask

    for stage in xrange(nrWeights -1):
      w = self.weights[stage]
      b = self.biases[stage]
      linearSum = T.dot(currentLayerValues, w) + b
      # dropout: give the next layer only some of the units from this layer
      if hiddenDropout in  [1.0, 1]:
        currentLayerValues = activationFunction.deterministic(linearSum)
      else:
        dropoutMaskHidden = self.theanoRng.binomial(n=1, p=hiddenDropout,
                                            size=linearSum.shape,
                                            dtype=theanoFloat)
        currentLayerValues = dropoutMaskHidden * activationFunction.deterministic(linearSum)

    # Last layer operations, no dropout in the output
    w = self.weights[nrWeights - 1]
    b = self.biases[nrWeights - 1]
    linearSum = T.dot(currentLayerValues, w) + b
    currentLayerValues = classificationActivationFunction.deterministic(linearSum)

    self.output = currentLayerValues

  def cost(self, y):
    return T.nnet.categorical_crossentropy(self.output, y)

class ClassifierBatch(object):

  def __init__(self, input, nrLayers, weights, biases,
               visibleDropout, hiddenDropout,
               activationFunction, classificationActivationFunction):

    self.input = input

    self.classificationWeights = classificationWeightsFromTestWeights(weights,
                                            visibleDropout=visibleDropout,
                                            hiddenDropout=hiddenDropout)

    nrWeights = nrLayers - 1

    currentLayerValues = input

    for stage in xrange(nrWeights -1):
      w = self.classificationWeights[stage]
      b = biases[stage]
      linearSum = T.dot(currentLayerValues, w) + b
      currentLayerValues = activationFunction.deterministic(linearSum)

    self.lastHiddenActivations = currentLayerValues

    w = self.classificationWeights[nrWeights - 1]
    b = biases[nrWeights - 1]
    linearSum = T.dot(currentLayerValues, w) + b
    currentLayerValues = classificationActivationFunction.deterministic(linearSum)

    self.output = currentLayerValues

  def cost(self, y):
    return T.nnet.categorical_crossentropy(self.output, y)


""" Class that implements a deep belief network, for classification """
class DBN(object):

  """
  Arguments:
    nrLayers: the number of layers of the network. In case of discriminative
        traning, also contains the classifcation layer
        type: integer
    layerSizes: the sizes of the individual layers.
        type: list of integers of size nrLayers
  """
  def __init__(self, nrLayers, layerSizes,
                binary,
                activationFunction=Sigmoid(),
                rbmActivationFunctionVisible=Sigmoid(),
                rbmActivationFunctionHidden=Sigmoid(),
                classificationActivationFunction=Softmax(),
                unsupervisedLearningRate=0.01,
                supervisedLearningRate=0.05,
                nesterovMomentum=True,
                rbmNesterovMomentum=True,
                momentumFactorForLearningRate=True,
                momentumMax=0.9,
                momentumForEpochFunction=getMomentumForEpochLinearIncrease,
                rmsprop=True,
                miniBatchSize=10,
                hiddenDropout=0.5,
                rbmHiddenDropout=0.5,
                visibleDropout=0.8,
                rbmVisibleDropout=1,
                weightDecayL1=0.0001,
                weightDecayL2=0.0001,
                sparsityConstraintRbm=False,
                sparsityRegularizationRbm=None,
                sparsityTragetRbm=None,
                preTrainEpochs=1,
                nameDataset=''):
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
    self.rbmNesterovMomentum = rbmNesterovMomentum
    self.rmsprop = rmsprop
    self.weightDecayL1 = weightDecayL1
    self.weightDecayL2 = weightDecayL2
    self.preTrainEpochs = preTrainEpochs
    self.activationFunction = activationFunction
    self.rbmActivationFunctionHidden = rbmActivationFunctionHidden
    self.rbmActivationFunctionVisible = rbmActivationFunctionVisible
    self.classificationActivationFunction = classificationActivationFunction
    self.momentumFactorForLearningRate = momentumFactorForLearningRate
    self.momentumMax = momentumMax
    self.momentumForEpochFunction = momentumForEpochFunction
    self.binary = binary

    self.sparsityRegularizationRbm = sparsityRegularizationRbm
    self.sparsityConstraintRbm = sparsityConstraintRbm
    self.sparsityTragetRbm = sparsityTragetRbm


    self.nameDataset = nameDataset

    print "hidden dropout in DBN", hiddenDropout
    print "visible dropout in DBN", visibleDropout

  def pretrain(self, data, unsupervisedData):
    nrRbms = self.nrLayers - 2

    self.weights = []
    self.biases = []
    self.generativeBiases = []

    currentData = data

    if unsupervisedData is not None:
      print "adding unsupervisedData"
      currentData = np.vstack((currentData, unsupervisedData))

    print "pre-training with a data set of size", len(currentData)

    lastRbmBiases = None
    lastRbmTrainWeights = None

    # TODO: refactor: make this dropout list more top level and reuse it
    dropoutList = [self.visibleDropout] + [self.hiddenDropout] * (self.nrLayers -1)

    for i in xrange(nrRbms):
      # If the network can be initialized from the previous one,
      # do so, by using the transpose of the already trained net
      if i > 0 and self.layerSizes[i+1] == self.layerSizes[i-1]:
        initialWeights = lastRbmTrainWeights.T
        initialBiases = lastRbmBiases
      else:
        initialWeights = None
        initialBiases = None

      if i == 0:
        unsupervisedLearningRate = self.unsupervisedLearningRate * 10
      else:
        unsupervisedLearningRate = self.unsupervisedLearningRate

      net = rbm.RBM(self.layerSizes[i], self.layerSizes[i+1],
                      learningRate=unsupervisedLearningRate,
                      visibleActivationFunction=self.rbmActivationFunctionVisible,
                      hiddenActivationFunction=self.rbmActivationFunctionHidden,
                      hiddenDropout=self.rbmHiddenDropout,
                      visibleDropout=self.rbmVisibleDropout,
                      rmsprop=True, # TODO: argument here as well?
                      nesterov=self.rbmNesterovMomentum,
                      initialWeights=initialWeights,
                      initialBiases=initialBiases,
                      trainingEpochs=self.preTrainEpochs,
                      sparsityConstraint=self.sparsityConstraintRbm,
                      sparsityTraget=self.sparsityTragetRbm,
                      sparsityRegularization=self.sparsityRegularizationRbm)

      net.train(currentData)

      # Use the test weights from the rbm, the ones the correspond to the incoming
      # weights for the hidden units
      # Then you have to divide by the dropout
      self.weights += [net.testWeights[1] / dropoutList[i]]
      # Only add the biases for the hidden unit
      b = net.biases[1]
      lastRbmBiases = net.biases
      # Do not take the test weight, take the training ones
      # because you will continue training with them
      lastRbmTrainWeights = net.weights
      self.biases += [b]
      self.generativeBiases += [net.biases[0]]

      # Let's update the current representation given to the next RBM
      currentData = net.hiddenRepresentation(currentData)

      # Average activation
      print "average activation after rbm pretraining"
      print currentData.mean()

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

  """
    Choose a percentage (percentValidation) of the data given to be
    validation data, used for early stopping of the model.
  """
  def train(self, data, labels, maxEpochs, validation=True, percentValidation=0.05,
            unsupervisedData=None, trainingIndices=None):

    # Required if the user wants to record on what indices they tested the dataset on
    self.trainingIndices = trainingIndices

    # Do a small check to see if the data is in between (0, 1)
    # if we claim we have binary stochastic units
    if self.binary:
      mins = data.min(axis=1)
      maxs = data.max(axis=1)
      assert np.all(mins >=0.0) and np.all(maxs < 1.0 + 1e-8)
    else:
      # We are using gaussian visible units so we need to scale the data
      # TODO: NO: pass in a scale argument
      if isinstance(self.rbmActivationFunctionVisible, Identity):
        print "scaling input data"
        data = scale(data)


      if unsupervisedData is not None:
        mins = unsupervisedData.min(axis=1)
        maxs = unsupervisedData.max(axis=1)
        assert np.all(mins) >=0.0 and np.all(maxs) < 1.0 + 1e-8

    print "shuffling training data"
    data, labels = shuffle(data, labels)

    if validation:
      nrInstances = len(data)
      validationIndices = np.random.choice(xrange(nrInstances),
                                           percentValidation * nrInstances)
      trainingIndices = list(set(xrange(nrInstances)) - set(validationIndices))
      trainingData = data[trainingIndices, :]
      trainingLabels = labels[trainingIndices, :]

      validationData = data[validationIndices, :]
      validationLabels = labels[validationIndices, :]

      self._trainWithGivenValidationSet(trainingData, trainingLabels,
                                       validationData, validationLabels, maxEpochs,
                                       unsupervisedData)
    else:
      trainingData = data
      trainingLabels = labels
      self.trainNoValidation(trainingData, trainingLabels, maxEpochs,
                                       unsupervisedData)

  #TODO: if this is used from outside, you have to scale the data as well
  # and also the validation data
  # Could be a good idea to use validation data from a different set?
  def _trainWithGivenValidationSet(self, data, labels,
                                  validationData,
                                  validationLabels,
                                  maxEpochs,
                                  unsupervisedData=None):

    sharedData = theano.shared(np.asarray(data, dtype=theanoFloat))
    sharedLabels = theano.shared(np.asarray(labels, dtype=theanoFloat))


    self.pretrain(data, unsupervisedData)

    self.nrMiniBatchesTrain = len(data) / self.miniBatchSize

    self.miniBatchValidateSize = min(len(validationData), self.miniBatchSize * 10)
    self.nrMiniBatchesValidate =  self.miniBatchValidateSize / self.miniBatchValidateSize

    sharedValidationData = theano.shared(np.asarray(validationData, dtype=theanoFloat))
    sharedValidationLabels = theano.shared(np.asarray(validationLabels, dtype=theanoFloat))
    # Does backprop for the data and a the end sets the weights
    self.fineTune(sharedData, sharedLabels, True,
                  sharedValidationData, sharedValidationLabels, maxEpochs)

  def trainNoValidation(self, data, labels, maxEpochs, unsupervisedData):
    sharedData = theano.shared(np.asarray(data, dtype=theanoFloat))
    sharedLabels = theano.shared(np.asarray(labels, dtype=theanoFloat))

    self.pretrain(data, unsupervisedData)

    self.nrMiniBatchesTrain = len(data) / self.miniBatchSize

    # Does backprop for the data and a the end sets the weights
    self.fineTune(sharedData, sharedLabels, False, None, None, maxEpochs)


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
  def fineTune(self, data, labels, validation, validationData, validationLabels,
               maxEpochs):
    print "supervisedLearningRate"
    print self.supervisedLearningRate
    batchLearningRate = self.supervisedLearningRate / self.miniBatchSize
    batchLearningRate = np.float32(batchLearningRate)

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
                                    activationFunction=self.activationFunction,
                                    classificationActivationFunction=self.classificationActivationFunction,
                                    initialWeights=self.weights,
                                    initialBiases=self.biases,
                                    visibleDropout=self.visibleDropout,
                                    hiddenDropout=self.hiddenDropout)

    classifier = ClassifierBatch(input=x, nrLayers=self.nrLayers,
                                 activationFunction=self.activationFunction,
                                 classificationActivationFunction=self.classificationActivationFunction,
                                 visibleDropout=self.visibleDropout,
                                 hiddenDropout=self.hiddenDropout,
                                 weights=batchTrainer.weights,
                                 biases=batchTrainer.biases)

    # TODO: remove training error from this
    # the error is the sum of the errors in the individual cases
    trainingError = T.sum(batchTrainer.cost(y))
    # also add some regularization costs
    error = trainingError
    for w in batchTrainer.weights:
      error+= self.weightDecayL1 * T.sum(abs(w)) + self.weightDecayL2 * T.sum(w ** 2)

    if DEBUG:
      mode = theano.compile.MonitorMode(post_func=detect_nan).excluding(
                                        'local_elemwise_fusion', 'inplace')
    else:
      mode = None

    if self.nesterovMomentum:
      preDeltaUpdates, updates = self.buildUpdatesNesterov(batchTrainer, momentum,
                    batchLearningRate, error)
      updateParamsWithMomentum = theano.function(
          inputs=[momentum],
          outputs=[],
          updates=preDeltaUpdates,
          mode = mode)

      updateParamsWithGradient = theano.function(
          inputs =[miniBatchIndex, momentum],
          outputs=trainingError,
          updates=updates,
          givens={
              x: data[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize],
              y: labels[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize]},
          mode=mode)

      def trainModel(miniBatchIndex, momentum):
        updateParamsWithMomentum(momentum)
        return updateParamsWithGradient(miniBatchIndex, momentum)
    else:

      updates = self.buildUpdatesSimpleMomentum(batchTrainer, momentum,
                    batchLearningRate, error)
      trainModel = theano.function(
            inputs=[miniBatchIndex, momentum],
            outputs=trainingError,
            updates=updates,
            givens={
                x: data[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize],
                y: labels[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize]})

      theano.printing.pydotprint(trainModel)

    trainingErrorNoDropout = theano.function(
          inputs=[miniBatchIndex],
          outputs=T.mean(classifier.cost(y)),
          givens={
              x: data[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize],
              y: labels[miniBatchIndex * self.miniBatchSize:(miniBatchIndex + 1) * self.miniBatchSize]})

    if validation:
    # Let's create the function that validates the model!
      validateModel = theano.function(inputs=[miniBatchIndex],
        outputs=T.mean(classifier.cost(y)),
        givens={
          x: validationData[miniBatchIndex * self.miniBatchValidateSize:(miniBatchIndex + 1) * self.miniBatchValidateSize],
          y: validationLabels[miniBatchIndex * self.miniBatchValidateSize:(miniBatchIndex + 1) * self.miniBatchValidateSize]})

      self.trainModelPatience(trainModel, validateModel, maxEpochs, trainingErrorNoDropout)
    else:
      if validationData is not None or validationLabels is not None:
        raise Exception(("You provided validation data but requested a train method "
                        "that does not need validation"))

      self.trainLoopModelFixedEpochs(batchTrainer, trainModel, maxEpochs)

    # Set up the weights in the dbn object
    self.x = x
    self.classifier = classifier

    self.weights = map(lambda x: x.get_value(), batchTrainer.weights)
    self.biases = map(lambda x: x.get_value(), batchTrainer.biases)

    self.classificationWeights = classificationWeightsFromTestWeights(self.weights,
                                      visibleDropout=self.visibleDropout,
                                      hiddenDropout=self.hiddenDropout)


  def trainLoopModelFixedEpochs(self, batchTrainer, trainModel, maxEpochs):
    trainingErrors = []

    try:
      for epoch in xrange(maxEpochs):
        print "epoch " + str(epoch)

        momentum = self.momentumForEpochFunction(self.momentumMax, epoch)

        for batchNr in xrange(self.nrMiniBatchesTrain):
          trainError = trainModel(batchNr, momentum) / self.miniBatchSize
          trainingErrors += [trainError]
    except KeyboardInterrupt:
      print "you have interrupted training"
      print "we will continue testing with the state of the network as it is"


    # plotTraningError(trainingErrors)

    print "number of epochs"
    print epoch


  def trainLoopWithValidation(self, trainModel, validateModel, maxEpochs):
    lastValidationError = np.inf
    count = 0
    epoch = 0

    validationErrors = []
    trainingErrors = []

    try:
      while epoch < maxEpochs and count < 8:
        print "epoch " + str(epoch)

        momentum = self.momentumForEpochFunction(self.momentumMax, epoch)

        for batchNr in xrange(self.nrMiniBatchesTrain):
          trainingErrorBatch = trainModel(batchNr, momentum) / self.miniBatchSize

        trainingErrors += [trainingErrorBatch]

        meanValidations = map(validateModel, xrange(self.nrMiniBatchesValidate))
        meanValidation = sum(meanValidations) / len(meanValidations)
        validationErrors += [meanValidation]

        if meanValidation > lastValidationError:
            count +=1
        else:
            count = 0
        lastValidationError = meanValidation

        epoch +=1
    except KeyboardInterrupt:
      print "you have interrupted training"
      print "we will continue testing with the state of the network as it is"

    plotTrainingAndValidationErros(trainingErrors, validationErrors)

    print "number of epochs"
    print epoch



  # A very greedy approach to training
  # A more mild version would be to actually take 3 conescutive ones
  # that give the best average (to ensure you are not in a luck place)
  # and take the best of them
  def trainModelGetBestWeights(self, batchTrainer, trainModel, validateModel, maxEpochs):
    bestValidationError = np.inf

    validationErrors = []
    trainingErrors = []


    bestWeights = None
    bestBiases = None
    bestEpoch = 0

    for epoch in xrange(maxEpochs):
      print "epoch " + str(epoch)

      momentum = self.momentumForEpochFunction(self.momentumMax, epoch)

      for batchNr in xrange(self.nrMiniBatchesTrain):
        trainingErrorBatch = trainModel(batchNr, momentum) / self.miniBatchSize

      trainingErrors += [trainingErrorBatch]

      meanValidations = map(validateModel, xrange(self.nrMiniBatchesValidate))
      meanValidation = sum(meanValidations) / len(meanValidations)

      validationErrors += [meanValidation]

      if meanValidation < bestValidationError:
        bestValidationError = meanValidation
        # Save the weights which are the best ones
        bestWeights = batchTrainer.weights
        bestBiases = batchTrainer.biases
        bestEpoch = epoch

    # If we have improved at all during training
    # not sure if things work well like this with theano stuff
    # maybe I need an update
    if bestWeights is not None and bestBiases is not None:
      batchTrainer.weights = bestWeights
      batchTrainer.biases = bestBiases

    plotTrainingAndValidationErros(trainingErrors, validationErrors)

    print "number of epochs"
    print epoch

    print "best epoch"
    print bestEpoch


  def trainModelPatience(self, trainModel, validateModel, maxEpochs, trainNoDropout):
    bestValidationError = np.inf
    epoch = 0
    doneTraining = False
    patience = 10 * self.nrMiniBatchesTrain # do at least 10 passes trough the data no matter what

    validationErrors = []
    trainingErrors = []
    trainingErrorNoDropout = []

    try:
      # while (epoch < maxEpochs) and not doneTraining:
      while (epoch < maxEpochs):
        # Train the net with all data
        print "epoch " + str(epoch)

        momentum = self.momentumForEpochFunction(self.momentumMax, epoch)

        for batchNr in xrange(self.nrMiniBatchesTrain):
          iteration = epoch * self.nrMiniBatchesTrain  + batchNr
          trainingErrorBatch = trainModel(batchNr, momentum) / self.miniBatchSize

          meanValidations = map(validateModel, xrange(self.nrMiniBatchesValidate))
          meanValidation = sum(meanValidations) / len(meanValidations)


          if meanValidation < bestValidationError:
            # If we have improved well enough, then increase the patience
            if meanValidation < bestValidationError:
              print "increasing patience"
              patience = max(patience, iteration * 2)

            bestValidationError = meanValidation

        validationErrors += [meanValidation]
        trainingErrors += [trainingErrorBatch]
        trainingErrorNoDropout +=  [trainNoDropout(batchNr)]

        if patience <= iteration:
          doneTraining = True

        epoch += 1
    except KeyboardInterrupt:
      print "you have interrupted training"
      print "we will continue testing with the state of the network as it is"

    plotTrainingAndValidationErros(trainingErrors, validationErrors)

    print "number of epochs"
    print epoch


  def buildUpdatesNesterov(self, batchTrainer, momentum,
                  batchLearningRate, error):

    if self.momentumFactorForLearningRate:
      lrFactor = 1.0 - momentum
    else:
      lrFactor = 1.0

    preDeltaUpdates = []
    for param, oldUpdate in zip(batchTrainer.params, batchTrainer.oldUpdates):
      preDeltaUpdates.append((param, param + momentum * oldUpdate))

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    deltaParams = T.grad(error, batchTrainer.params)
    updates = []
    parametersTuples = zip(batchTrainer.params,
                           deltaParams,
                           batchTrainer.oldUpdates,
                           batchTrainer.oldMeanSquare)

    for param, delta, oldUpdate, oldMeanSquare in parametersTuples:
      if self.rmsprop:
        meanSquare = 0.9 * oldMeanSquare + 0.1 * delta ** 2
        paramUpdate = - lrFactor * batchLearningRate * delta / T.sqrt(meanSquare + 1e-8)
        updates.append((oldMeanSquare, meanSquare))
      else:
        paramUpdate = - lrFactor * batchLearningRate * delta

      newParam = param + paramUpdate

      updates.append((param, newParam))
      updates.append((oldUpdate, momentum * oldUpdate + paramUpdate))

    return preDeltaUpdates, updates

  def buildUpdatesSimpleMomentum(self, batchTrainer, momentum,
                  batchLearningRate, error):

    if self.momentumFactorForLearningRate:
      lrFactor = 1.0 - momentum
    else:
      lrFactor = 1.0

    deltaParams = T.grad(error, batchTrainer.params)
    updates = []
    parametersTuples = zip(batchTrainer.params,
                           deltaParams,
                           batchTrainer.oldUpdates,
                           batchTrainer.oldMeanSquare)

    for param, delta, oldUpdate, oldMeanSquare in parametersTuples:
      paramUpdate = momentum * oldUpdate
      if self.rmsprop:
        meanSquare = 0.9 * oldMeanSquare + 0.1 * delta ** 2
        paramUpdate += - lrFactor * batchLearningRate * delta / T.sqrt(meanSquare + 1e-8)
        updates.append((oldMeanSquare, meanSquare))
      else:
        paramUpdate += - lrFactor * batchLearningRate * delta

      newParam = param + paramUpdate

      updates.append((param, newParam))
      updates.append((oldUpdate, paramUpdate))

    return updates

  def classify(self, dataInstaces):
    dataInstacesConverted = theano.shared(np.asarray(dataInstaces, dtype=theanoFloat))

    classifyFunction = theano.function(
            inputs=[],
            outputs=self.classifier.output,
            updates={},
            givens={self.x: dataInstacesConverted}
            )
    lastLayers = classifyFunction()
    return lastLayers, np.argmax(lastLayers, axis=1)


  """The speed of this function could be improved but since it is never called
  during training and it is for illustrative purposes that should not be a problem. """
  def sample(self, nrSamples):
    nrRbms = self.nrLayers - 2

    # Create a random samples of the size of the last layer
    if self.binary:
      samples = np.random.rand(nrSamples, self.layerSizes[-2])
    else:
      samples = np.random.randint(255, size=(nrSamples, self.layerSizes[-2]))

    # You have to do it  in decreasing order
    for i in xrange(nrRbms -1, 0, -1):
      # If the network can be initialized from the previous one,
      # do so, by using the transpose of the already trained net

      weigths = self.classificationWeights[i-1].T
      biases = np.array([self.biases[i-1], self.generativeBiases[i-1]])
      net = rbm.RBM(self.layerSizes[i], self.layerSizes[i-1],
                      learningRate=self.unsupervisedLearningRate,
                      visibleActivationFunction=self.rbmActivationFunctionVisible,
                      hiddenActivationFunction=self.rbmActivationFunctionHidden,
                      hiddenDropout=1.0,
                      visibleDropout=1.0,
                      rmsprop=True, # TODO: argument here as well?
                      nesterov=self.rbmNesterovMomentum,
                      initialWeights=weigths,
                      initialBiases=biases)

      # Do 20 layers of gibbs sampling for the last layer
      print samples.shape
      print biases.shape
      print biases[1].shape
      if i == nrRbms - 1:
        samples = net.reconstruct(samples, cdSteps=20)

      # Do pass trough the net
      samples = net.hiddenRepresentation(samples)

    return samples

  """The speed of this function could be improved but since it is never called
  during training and it is for illustrative purposes that should not be a problem. """
  def getHiddenActivations(self, data):
    nrRbms = self.nrLayers - 2

    activations = data
    activationsList = []

    # You have to do it  in decreasing order
    for i in xrange(nrRbms):
      # If the network can be initialized from the previous one,
      # do so, by using the transpose of the already trained net
      weigths = self.classificationWeights[i]
      biases = np.array([self.generativeBiases[i], self.biases[i]])
      net = rbm.RBM(self.layerSizes[i], self.layerSizes[i+1],
                      learningRate=self.unsupervisedLearningRate,
                      visibleActivationFunction=self.rbmActivationFunctionVisible,
                      hiddenActivationFunction=self.rbmActivationFunctionHidden,
                      hiddenDropout=1.0,
                      visibleDropout=1.0,
                      rmsprop=True, # TODO: argument here as well?
                      nesterov=self.rbmNesterovMomentum,
                      initialWeights=weigths,
                      initialBiases=biases)

      # Do pass trough the net
      activations = net.hiddenRepresentation(activations)
      activationsList += [activations]

    return activationsList


  def hiddenActivations(self, data):
    dataInstacesConverted = theano.shared(np.asarray(data, dtype=theanoFloat))

    classifyFunction = theano.function(
              inputs=[],
              outputs=self.classifier.output,
              updates={},
              givens={self.x: dataInstacesConverted})

    classifyFunction()

    return self.classifier.lastHiddenActivations

def classificationWeightsFromTestWeights(weights, visibleDropout, hiddenDropout):
  classificationWeights = [visibleDropout * weights[0]]
  classificationWeights += map(lambda x: x * hiddenDropout, weights[1:])

  return classificationWeights
