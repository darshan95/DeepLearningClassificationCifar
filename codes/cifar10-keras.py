"Author: Nishant Prateek"

from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint
import cPickle as pickle
from sklearn.metrics import confusion_matrix


"""
A simple convnet for object recognition using cifar10 dataset. The convnet take raw features as input.
The model gives 80.68 per cent accuracy after training for 100 epochs.

Confusion matrix:
[[868  12  25  14   8   2   6   6  40  19]
 [ 11 915   2   6   1   1   5   1  15  43]
 [ 60   4 703  53  57  45  37  21  14   6]
 [ 16   6  38 650  50 140  46  29  15  10]
 [ 21   4  48  56 760  30  28  44   5   4]
 [ 14   2  31 143  19 742  11  31   4   3]
 [  5   5  32  54  28  18 846   6   3   3]
 [ 18   0  19  33  30  43   1 847   3   6]
 [ 64  15   7   9   2   4   4   3 875  17]
 [ 29  61   4   8   3   3   3   9  18 862]]

Each epoch during training runs for about 5 minutes on NVidia GeForce GT 630M GPU

To execute, run: THEANO_FLAGS=device=gpu,floatX=float32 python cifar10-keras.py 

 """


batch_size = 50
nb_classes = 10
nb_epoch = 100
data_augmentation = False

img_rows, img_cols = 32, 32
img_channels = 3

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print ("binary classes done...")

#intializing models
model = Sequential()

#first layer convolution -> relu -> maxpool
model.add(Convolution2D(32, 3, 3, border_mode='full',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#second layer convolution -> relu -> maxpool
model.add(Convolution2D(64, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# fully connected layer + softmax classifier
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#optimization
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

json_string = model.to_json()
open('cifar10-cnn.json', 'w').write(json_string)

print ("model built...")

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train = X_train/255
X_test = X_test/255

#saving weights after each epoch
checkpointer = ModelCheckpoint(filepath="weights-cifar10/weights.{epoch:02d}.hdf5", verbose=1, save_best_only=False)

print("training starts...")
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[checkpointer])

#testing and accuracy
print ("testing time...")
score = model.evaluate(X_test, Y_test, batch_size=batch_size, show_accuracy = True)

print('Test score:', score)
classes = model.predict_classes(X_test, batch_size=32)
pickle.dump(classes, open('cifar10-test.pkl', 'wb'))

#confusion matrix
conf_mat = confusion_matrix(y_test, classes)
print (conf_mat)
pickle.dump(conf_mat, open('conf_mat-cifar10.pkl', 'wb'))