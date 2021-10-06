# Adam Rilatt
# 09 / 24 / 20
# Ultimate Tic Tac Toe Neural Net

from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
import random
import h5py


''' ============================================================================
                                USER PARAMETERS
    ========================================================================='''

# where to save this model to.
MODEL_LOCATION = 'two_layer_wonder.h5'

# location of training dataset.
RECORD_FILE = h5py.File('tictac_record_test.h5', 'r')

# select what amount of data (out of 1.0) goes to testing. 0.1 to 0.3 is good.
TEST_TRAIN = 0.2

# specify the number of logical CPU cores available for training process. my
# device has 6 physical / 12 logical cores.
NUM_CORES = 12

# whether or not to use GPU acceleration. my device has a 1660ti with 1.5k CUDA
# cores. depending on your card, mileage may vary.
GPU_TRAIN = True

# initializes RNG for constancy between tests.
SEED = 451
np.random.seed(SEED)

# number of epochs to train the model. select epoch number based on point of
# diminishing returns to avoid under- or over-fitting.
NUM_EPOCHS = 20

# specifies the batch size during training. higher batch sizes train faster,
# but model may suffer in generalization. powers of 2 preferred.
BATCH_SIZE = 48


''' ============================================================================
                        TRAINING CONFIGURATION & SETUP
    ========================================================================='''

import tensorflow.compat.v1 as tf
import tensorflow.python.keras.backend as K

num_CPU = 1
if GPU_TRAIN:
    num_GPU = 1
else:
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads = NUM_CORES,
                        inter_op_parallelism_threads = NUM_CORES,
                        allow_soft_placement = True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU}
                       )

session = tf.Session(config = config)
K.set_session(session)


''' ============================================================================
                                DATA PREPROCESSING
    ========================================================================='''

# some of the samples recorded strangely, so instead of tossing out the entire
# datafile, we use only the x-y points that have a corresponding pair. it's
# better than nothing.
y = RECORD_FILE['Y'][:400_000]
x = RECORD_FILE['X'][:len(y)]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = TEST_TRAIN,
                                                    random_state = SEED)


''' ============================================================================
                            MODEL CREATION & TRAINING
    ========================================================================='''

# Softmax model format that seems to do okay...
#brain = Sequential()
#brain.add(Dense(192, input_shape = (81,), activation = 'relu'))
#brain.add(Dense(256, activation = 'relu'))
#...
#brain.add(Dense(3,   activation = 'softmax'))
#brain.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#brain.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 30, batch_size = 48, shuffle = True)

brain = Sequential()
brain.add(Dense(81, input_shape = (81,), activation = 'relu'))
brain.add(Dense(9, activation = 'relu'))
brain.add(Dense(3, activation = 'softmax'))

brain.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                                                 metrics = ['accuracy'])

brain.fit(X_train, Y_train, validation_data = (X_test, Y_test),
                                     epochs = NUM_EPOCHS,
                                 batch_size = BATCH_SIZE,
                                    shuffle = True
         )

scores = brain.evaluate(X_test, Y_test)
print(scores)

print("Saving to %s..." % MODEL_LOCATION)
brain.save(MODEL_LOCATION)
print("... Saved.")
