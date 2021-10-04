###########################################
## Gianmarco Mulazzani 2021 - MSc Thesis ##
## 06 - Deep Learning applied to LOB Data #
###########################################

# This code is used to prepare the dataset and the used what is produced to run the DeepLOB Model
# It is mainly based on Github resources published by Zhang et al. Notice that some arrangements
# have been made because the FI-2010 dataset has LOB states on the rows and time obs on columns, in
# our case the equivalent matrix is transposed.

import numpy as np
from Preparation_Functions import prepare_x, prepare_x_y

# Attention line 15-16 could generate an error due to the 0.8 factor. But the error only
# appears because the dimension of the sample csv file is very limited.

dec_data = np.loadtxt('Data_Normalised_PFE.N.txt')
dec_train = dec_data[:int(np.floor(dec_data.shape[0] * 0.8)), :]
dec_val = dec_data[int(np.floor(dec_data.shape[0] * 0.8)):, :]

# Here they use the test datasets. Note that I do not have already this separation and so 
# I could write 

#dec_test1 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_7.txt')
#dec_test2 = np.loadtxt('Test_Dst_NoAuction_DecPre_CF_8.txt')
dec_test3 = np.loadtxt('Data_Normalised_WMT.N.txt')
#dec_test = np.hstack((dec_test1, dec_test2, dec_test3)) # This is a way to concatenate arrays
dec_test = np.hstack((dec_test3)) # This is a way to concatenate arrays

k = 2 # which prediction horizon
T = 100 # the length of a single input
n_hiddens = 64
checkpoint_filepath = './model_tensorflow2/weights' # ?

trainX_CNN, trainY_CNN = prepare_x_y(dec_train, k, T)
valX_CNN, valY_CNN = prepare_x_y(dec_val, k, T)
#testX_CNN, testY_CNN = prepare_x_y(dec_test, k, T)

print(trainX_CNN[1,1])

print(trainX_CNN.shape, trainY_CNN.shape)
print(valX_CNN.shape, valY_CNN.shape)
#print(testX_CNN.shape, testY_CNN.shape)