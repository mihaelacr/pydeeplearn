language: PYTHON
name:     "adversarial"

variable {
 name: "supervisedLearningRate"
 type: FLOAT
 size: 1
 min:  0.00001
 max:  1.0
}

variable {
 name: "unsupervisedLearningRate"
 type: FLOAT
 size: 1
 min:  0.00001
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
