'''Generael constants for saddle-model, that can be used not just (but also) in training'''

import numpy as np

QUERY_SIZES = {"S":[0.01*np.pi, 0.3*np.pi], "M":[0.3*np.pi, 1.0*np.pi], "L":[1.0*np.pi, 2*np.pi]} #Lower and upper value for lower and upper max_pi value
MLFLOW_HOST = "127.0.0.1"
MLFLOW_PORT = "8080"
N_DIMS = 3 
N_BITS = 4
N_POINTS_PER_AXIS = 2**N_BITS
N_GRID_POINTS     = (N_POINTS_PER_AXIS)**N_DIMS
