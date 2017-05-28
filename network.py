import csv
import cv2
import numpy as np
import os
import sklearn

#sample data with additional recorded turns
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.3)

# generator function for dynamic data loading
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])

                #center image
                images.append(center_image)
                angles.append(center_angle)
                #flip center image
                images.append(np.fliplr(cv2.imread(name)))
                angles.append(center_angle * -1)

                #add images from left and right camera with additional offsets
                name = batch_sample[1]
                left_image = cv2.imread(name)
                left_angle = center_angle + 0.15

                name = batch_sample[2]
                right_image = cv2.imread(name)
                right_angle = center_angle - 0.15

                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Activation, MaxPooling2D, Dropout, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3))) #normalization
model.add(Cropping2D(cropping=((55,20), (0,0)))) #crop image to only leave thee road
model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)
model.save('model.h5')

import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show() #show loses