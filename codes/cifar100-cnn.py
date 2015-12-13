from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint
import cPickle as pickle
from sklearn.metrics import confusion_matrix


batch_size = 50
nb_classes = 20
nb_epoch = 25
img_rows, img_cols = 32, 32
img_channels = 3

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='coarse')
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print (Y_test.shape)
print ("binary classes done...")

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

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

print ("model built...")

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train = X_train/255
X_test = X_test/255

#saving weights after each epoch
checkpointer = ModelCheckpoint(filepath="weights-cifar100/weights.{epoch:02d}.hdf5", verbose=1, save_best_only=False)

print("training starts...")
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[checkpointer])

#testing and accuracy
print ("testing time...")
score = model.evaluate(X_test, Y_test, batch_size=batch_size, show_accuracy = True)

print('Test score:', score)
classes = model.predict_classes(X_test, batch_size=32)
pickle.dump(classes, open('cifar100-test.pkl', 'wb'))

#confusion matrix
conf_mat = confusion_matrix(y_test, classes)
print (conf_mat)
pickle.dump(conf_mat, open('conf_mat-cifar100.pkl', 'wb'))