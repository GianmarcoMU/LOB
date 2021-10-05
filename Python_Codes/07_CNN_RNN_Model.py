###########################################
## Gianmarco Mulazzani 2021 - MSc Thesis ##
## 07 - Deep Learning applied to LOB Data #
###########################################

# This code is used to implement the DeepLOB Model.
# It is mainly based on Github resources published by Zhang et al. 
# Notice that some arrangements have been made to adapt the code to our dataset.

import pandas as pd
import pickle
import numpy as np
import keras
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, LSTM, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
import tensorflow as tf
from Preparation_Functions import prepare_x, prepare_x_y

from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# set random seeds
np.random.seed(1)
tf.random.set_seed(2)

# Attention line 15-16 could generate an error due to the 0.8 factor. But the error only
# appears because the dimension of the sample csv file is very limited. Moreover, the factor
# 0.8 should be change according to the dimension of the validation set desired.

# Since the output of previous codes is a txt file for each instrument, I need to merge them.

# dec_data_JPM = np.loadtxt('Data_Normalised_Train_JPM.N.txt')
# dec_data_JNJ = np.loadtxt('Data_Normalised_Train_JNJ.N.txt')
# dec_data_BRK = np.loadtxt('Data_Normalised_Train_BRK.N.txt')
# dec_data_PFE = np.loadtxt('Data_Normalised_Train_PFE.N.txt')
# dec_data_WMT = np.loadtxt('Data_Normalised_Train_WMT.N.txt')
# dec_data = np.hstack((dec_data_JPM, dec_data_JNJ, dec_data_BRK, dec_data_PFE, dec_data_WMT))

dec_data = np.loadtxt('Data_Normalised_Train_PFE.N.txt')
dec_train = dec_data[:int(np.floor(dec_data.shape[0] * 0.8)), :]
dec_val = dec_data[int(np.floor(dec_data.shape[0] * 0.8)):, :]

# dec_test_JPM = np.loadtxt('Data_Normalised__Test_JPM.N.txt')
# dec_test_JNJ = np.loadtxt('Data_Normalised__Test_JNJ.N.txt')
# dec_test_BRK = np.loadtxt('Data_Normalised__Test_BRK.N.txt')
# dec_test_PFE = np.loadtxt('Data_Normalised__Test_PFE.N.txt')
# dec_test_WMT = np.loadtxt('Data_Normalised__Test_WMT.N.txt')
# dec_test = np.hstack((dec_test_JPM, dec_test_JNJ, dec_test_BRK, dec_test_PFE, dec_test_WMT)) 

dec_test = np.loadtxt('Data_Normalised__Test_PFE.N.txt')

k = 2 # which prediction horizon
T = 100 # the length of a single input
n_hiddens = 64
checkpoint_filepath = './model_tensorflow2/weights'

trainX_CNN, trainY_CNN = prepare_x_y(dec_train, k, T)
valX_CNN, valY_CNN = prepare_x_y(dec_val, k, T)
testX_CNN, testY_CNN = prepare_x_y(dec_test, k, T)

print(trainX_CNN.shape, trainY_CNN.shape)
print(valX_CNN.shape, valY_CNN.shape)
print(testX_CNN.shape, testY_CNN.shape)

# Next, we define the model function.

def create_deeplob(T, NF, number_of_lstm):
    input_lmd = Input(shape=(T, NF, 1))
    
    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)
    
    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)
    conv_reshape = keras.layers.Dropout(0.2, noise_shape=(None, 1, int(conv_reshape.shape[2])))(conv_reshape, training=True)

 # build the last LSTM layer
    conv_lstm = LSTM(number_of_lstm)(conv_reshape)

    # build the output layer
    out = Dense(3, activation='softmax')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

deeplob = create_deeplob(trainX_CNN.shape[1], trainX_CNN.shape[2], n_hiddens)
deeplob.summary()

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)

# Here, we fit the model and then test it on the test dataset, printing results.

deeplob.fit(trainX_CNN, trainY_CNN, validation_data=(valX_CNN, valY_CNN), 
            epochs=200, batch_size=128, verbose=2, callbacks=[model_checkpoint_callback])

deeplob.load_weights(checkpoint_filepath)
pred = deeplob.predict(testX_CNN)

print('accuracy_score:', accuracy_score(np.argmax(testY_CNN, axis=1), np.argmax(pred, axis=1)))
print(classification_report(np.argmax(testY_CNN, axis=1), np.argmax(pred, axis=1), digits=4))