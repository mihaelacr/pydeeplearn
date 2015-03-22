language: PYTHON
name:     "dbnmnist"

variable {
 name: "supervisedLearningRate"
 type: FLOAT
 size: 1
 min:  0
 max:  1
}

variable {
 name: "unsupervisedLearningRate"
 type: FLOAT
 size: 1
 min:  0
 max:  1
}

variable {
 name: "momentumMax"
 type: FLOAT
 size: 1
 min:  0.5
 max:  1.0
}

variable {
 name: "hiddenDropout"
 type: FLOAT
 size: 1
 min:  0.5
 max:  1.0
}

variable {
 name: "visibleDropout"
 type: FLOAT
 size: 1
 min:  0.5
 max:  1.0
}

variable {
 name: "miniBatchSize"
 type: INT
 size: 1
 min:  10
 max:  1000
}

variable {
 name: "maxEpochs"
 type: INT
 size: 1
 min:  100
 max:  1000
}

# TODO(mihaelacr): add the number of pretraining epochs
# variable {
#  name: "preTrainEpochs"
#  type: INT
#  size: 1
#  min:  10
#  max:  1000
# }

