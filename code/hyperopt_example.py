# This module creates an optimization of hyper-parameters of a DBN using hyperopt library (https://github.com/hyperopt/hyperopt).
#  
# 

import numpy as np
import hyperopt

from hyperopt import hp, fmin, tpe
from sklearn import cross_validation
from lib import deepbelief as db
from read import readmnist
from lib.activationfunctions import *



space = ( 
	hp.qloguniform( 'l1_dim', log( 10 ), log( 1000 ), 1 ), 
	hp.qloguniform( 'l2_dim', log( 10 ), log( 1000 ), 1 ),
	hp.loguniform( 'learning_rate', log( 1e-5 ), log( 1e-2 )),
	hp.uniform( 'momentum', 0.5, 0.99 ),
	hp.uniform( 'l1_dropout', 0.1, 0.9 ),
	hp.uniform( 'decay_factor', 1 + 1e-3, 1 + 1e-1 )
)


best = fmin( run_test, space, algo = tpe.suggest, max_evals = 50 )

print best
