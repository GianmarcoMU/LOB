###########################################
## Gianmarco Mulazzani 2021 - MSc Thesis ##
## Deep Learning applied to LOB Data ######
###########################################

# This code is used to define some function used in code 07, which is the
# DeepLOB Model implementation. It is based on Github resources published by Zhang et al.

import numpy as np
from keras.utils import np_utils

# Note that we have made some arrangements to the original code. In fact, Zhang et al. started from
# FI-2010 dataset where features were stored in the rows while LOB states in the columns. Our dataset
# is exactly the transposed of it.

def prepare_x(data):
    df1 = data[:, :40]
    return np.array(df1)

def get_label(data):
    lob = data[:, -3:]
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)
    dY = np.array(Y)
    dataY = dY[T - 1:N]
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]
    return dataX.reshape(dataX.shape + (1,)), dataY

def prepare_x_y(data, k, T):
    x = prepare_x(data)
    y = get_label(data)
    x, y = data_classification(x, y, T=T)
    y = y[:,k] - 1
    y = np_utils.to_categorical(y, 3)
    return x, y