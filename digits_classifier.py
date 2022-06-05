# -*- coding: utf-8 -*-
"""
"""

import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import time
import os
from os import walk
import random

img_rows, img_cols = 28, 28
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A1", "A2", "B1", "B2", "C", "D1", "D2", "E1",
          "E2", "F1", "F2", "G1", "G2", "H1", "H2", "I", "J", "K", "L", "M", "N1", "N2", "O", "P", "Q1",
          "Q2", "R1", "R2", "S", "T1", "T2", "U", "V", "W", "X", "Y", "Z"]  # 1 for uppercase and 2 for lowercase

ALL = []
ALLLabel = []
for gt in range(len(labels)):
    mypath = "C:\\{}".format(labels[gt])
    os.chdir("C:\\{}".format(labels[gt]))

    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break

    for i in range(len(f)):
        image = cv2.imread("{}".format(f[i]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ALL.append(image)
        
        ALLLabel.append(gt)    # 26 classes (or 37, or 47)

# Randomly shuffling the lists before turning them into a numpy array
random.seed(8)
mapIndexPosition = list(zip(ALL, ALLLabel))
random.shuffle(mapIndexPosition)
ALL, ALLLabel = zip(*mapIndexPosition)

# Turning the lists into a numpy array
ALL = np.array(ALL)
ALLLabel = np.array(ALLLabel)

# Visualizing some images
'''plt.imshow(ALL[15], cmap = 'gray')
plt.title('Classe ' + str(ALLLabel[15]))

plt.imshow(X_treinamento[841], cmap = 'gray')
plt.title('Classe ' + str(y_treinamento[841]))'''

# Separating the total database into train and test according to the specified proportion
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(ALL, ALLLabel, test_size = 0.2)

# Continuing with the pre processing
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 47)
classe_teste = np_utils.to_categorical(y_teste, 47)

# CREATE MORE IMAGES VIA DATA AUGMENTATION
datagen = ImageDataGenerator(
        rotation_range = 5, 
        shear_range = 0.1,
        zoom_range = 0.2,  
        width_shift_range = 0.1, 
        height_shift_range = 0.1)

data_path = "C:\\"

classificador = Sequential()

classificador.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
classificador.add(BatchNormalization())
classificador.add(Conv2D(32, kernel_size = 3, activation='relu'))
classificador.add(BatchNormalization())
classificador.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
classificador.add(BatchNormalization())
classificador.add(Dropout(0.4))

classificador.add(Conv2D(64, kernel_size = 3, activation='relu'))
classificador.add(BatchNormalization())
classificador.add(Conv2D(64, kernel_size = 3, activation='relu'))
classificador.add(BatchNormalization())
classificador.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
classificador.add(BatchNormalization())
classificador.add(Dropout(0.4))

classificador.add(Conv2D(128, kernel_size = 4, activation='relu'))
classificador.add(BatchNormalization())
classificador.add(Flatten())
classificador.add(Dropout(0.4))
classificador.add(Dense(47, activation='softmax'))

# COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
classificador.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
checkpointer = ModelCheckpoint(filepath=data_path + "\\classifier-3-{epoch:02d}-{val_accuracy:.4f}.hdf5",
                               monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)
    
    
# DECREASE LEARNING RATE EACH EPOCH
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# TRAIN NETWORKS
epochs = 200
batch_size = 64

start = time.time()
    
history = classificador.fit_generator(datagen.flow(previsores_treinamento, classe_treinamento, batch_size),
    epochs = epochs, steps_per_epoch = previsores_treinamento.shape[0]//batch_size,  
    validation_data = (previsores_teste, classe_teste), callbacks = [annealer, checkpointer], verbose=1)
    
 
end = time.time()
print("Total time of processing: {} minutes".format((end-start)/60))


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1, figsize=(18, 10))
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

###################################################################################################
###################################################################################################

