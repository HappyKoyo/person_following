#!/usr/bin/env python

# General
import os
import numpy as np
import matplotlib.pyplot as plt

# Keras
import keras
from keras import layers,models,losses
from keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Reshape
from keras.layers.normalization import BatchNormalization

# Constant Definition
EPOCHS = 5 

# Initial Setting
model = models.Sequential()

# --- Model Description ---
# Conv1 84 -> 40
model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu', input_shape=(84,84,4)))
model.add(BatchNormalization())
# Conv2 40 -> 18
model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu'))
# Conv3 18 -> 7
model.add(Conv2D(64, kernel_size=5, strides=(2,2), activation='relu'))
# Flatten 7*7*64 -> 3136
model.add(Flatten())
print model.output_shape
model.add(Dense(128))
model.add(Dense(64))
# LSTM
model.add(Reshape((1,64)))
model.add(LSTM(64))
model.add(Reshape((1,64)))
model.add(LSTM(64))
# Dence2 -> 2
model.add(Dense(2))

# --- Optimize Manner ---
model.compile(optimizer='adam',
              loss=losses.mean_squared_error,
              metrics=['accuracy'])
keras.optimizers.Adam(lr=0.0003,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0)

# Loading Cross-Validation Set
val_color = np.load("../data/val/npy/color/1576647627.53.npy")
val_depth = np.load("../data/val/npy/depth/1576647627.53.npy")
val_joy   = np.load("../data/val/npy/joy/1576647627.53.npy")
# compose color and depth
val_depth = np.reshape(val_depth,(128,1,84,84))
val_rgbd  = np.append(val_color,val_depth,axis=1)
val_rgbd  = np.reshape(val_rgbd,(128,84,84,4))

# Loading Dataset
directory = os.listdir("../data/train/npy/color")
# History List
loss_hist = []
val_hist = []
hist_cnt = 0
# Optimizing Manner
for i in range(EPOCHS):
    for data in directory:
        hist_cnt += 1
        train_color = np.load("../data/train/npy/color/"+data)
        train_depth = np.load("../data/train/npy/depth/"+data)
        train_joy   = np.load("../data/train/npy/joy/"+data)
        # compose color and depth
        train_depth = np.reshape(train_depth,(128,1,84,84))
        train_rgbd  = np.append(train_color,train_depth,axis=1)
        train_rgbd  = np.reshape(train_rgbd,(128,84,84,4))
        #hist = model.fit(train_rgbd, train_joy,batch_size=1,verbose=1,epochs=1,validation_split=0.2)
        hist = model.fit(train_rgbd, train_joy,batch_size=1,verbose=1,epochs=1,validation_data=(val_rgbd,val_joy))
        loss_hist.append(hist.history["loss"][0])
        val_hist.append(hist.history["val_loss"][0])
        print hist.history["loss"]

# Display Trainig Loss
plt.plot(range(1,hist_cnt+1),loss_hist,label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()

# Display Validation Loss
plt.plot(range(1,hist_cnt+1),val_hist,label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()
