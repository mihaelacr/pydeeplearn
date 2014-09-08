import numpy as np

# TODO: does momentum make sense here
#  a momentum max rather
class TrainingOptions(object):

  def __init__(self, miniBatchSize,
        learningRate,
        momentumMax=0.0,
        rmsprop=False,
        nesterovMomentum=False,
        momentumFactorForLearningRate=False):
    self.miniBatchSize = miniBatchSize
    self.learningRate = learningRate
    self.momentumMax = np.float32(momentumMax)
    self.rmsprop = rmsprop
    self.nesterov = nesterovMomentum
    self.momentumFactorForLearningRate = momentumFactorForLearningRate
    self.batchLearningRate = np.float32(learningRate / miniBatchSize)
