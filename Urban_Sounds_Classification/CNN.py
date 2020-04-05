from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras import optimizers, models


def classify(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(1, 128, 345), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9),
              metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

    print('accuracy=', accuracy)

epochs = 25
learning_rate = 1e-4
batch_size = 32
x_train = np.expand_dims(np.load('preprocessed_data/updated/trainfeatures2d.npy'), axis=1)
y_train = np.load('preprocessed_data/updated/trainlabels2d.npy')
x_test = np.expand_dims(np.load('preprocessed_data/updated/testfeatures2d.npy'), axis=1)
y_test = np.load('preprocessed_data/updated/testlabels2d.npy')
classify(x_train, y_train, x_test, y_test)
