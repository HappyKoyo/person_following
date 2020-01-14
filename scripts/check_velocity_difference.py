#!/usr/bin/env python

# General
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# Keras
import keras
from keras import layers,models,losses
from keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape, LSTM
from keras.layers.normalization import BatchNormalization

WEIGHT_NAME = "1579002517.73_8.h5"

# --- Model Description ---
model = models.Sequential()
# Conv1 84 -> 40
model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu', input_shape=(84,84,4)))
#model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu', input_shape=(84,84,1)))

model.add(BatchNormalization())
# Conv2 40 -> 18
model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu'))
# Conv3 18 -> 7
model.add(Conv2D(64, kernel_size=5, strides=(2,2), activation='relu'))
# Flatten 7*7*64 -> 3136
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.3))
# LSTM
model.add(Reshape((1,64)))
model.add(LSTM(64))
model.add(Reshape((1,64)))
model.add(LSTM(64))
# Dence2 -> 2
#model.add(Dense(30))
model.add(Dense(2))
model.load_weights("../weights/"+WEIGHT_NAME)

VAL_DIR   = os.listdir("../data/train/color")

#for data in VAL_DIR:
#    val_data_list.append(data)
#random.shuffle(val_data_list)

for data in VAL_DIR:
    val_color = np.load("../data/train/color/"+data)
    val_depth = np.load("../data/train/depth/"+data)
    val_joy   = np.load("../data/train/joy/"+data)
    val_joy[:,1] = val_joy[:,1]/3
    print data
    val_depth = val_depth.reshape(512,84,84,1)
    val_rgbd = np.append(val_color,val_depth,axis=3)
    velocities = []
    val_data_list = []

    for i in range(0,512):
        rgbd = val_rgbd[i]
        rgbd = rgbd.reshape(1,84,84,4)
        vel = model.predict(rgbd)
        #depth = val_depth[i]
        #depth = depth.reshape(1,84,84,1)
        #vel = model.predict(depth)
        velocities.append(vel)
    

    velocities = np.array(velocities)
    #velocities = velocities.reshape(2,512)
    #val_joy = val_joy.reshape(2,512)
    print val_joy
    plt.plot(range(0,512),val_joy[:,0],label="teaching linear")
    print velocities.shape
    plt.plot(range(0,512),velocities[:,:,0],label="predicting linear")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()

    plt.plot(range(0,512),val_joy[:,1],label="teaching roll")
    plt.plot(range(0,512),velocities[:,:,1],label="predicting roll")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.show()

