#!/usr/bin/env python

# General
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Keras
import keras
from keras import layers,models,losses
from keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Reshape
from keras.layers.normalization import BatchNormalization

# Constant Definition
EPOCHS = 50 

# Initial Setting
model = models.Sequential()

# --- Model Description ---
# Conv1 84 -> 40
model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu', input_shape=(84,84,4)))
model.add(BatchNormalization())
# Conv2 40 -> 18
model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu'))
# Conv3 18 -> 7
#model.add(Conv2D(64, kernel_size=5, strides=(2,2), activation='relu'))
# Flatten 7*7*64 -> 3136
model.add(Flatten())
#model.add(Dense(128))
model.add(Dense(64))
# LSTM
model.add(Reshape((1,64)))
model.add(LSTM(64))
#model.add(Reshape((1,64)))
#model.add(LSTM(64))
# Dence2 -> 2
model.add(Dense(2))

# --- Optimize Manner ---
model.compile(optimizer='adam',
              loss=losses.mean_squared_error,
              metrics=['accuracy'])
keras.optimizers.Adam(lr=0.003,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0)


# --- Optimize Model ---
train_dir = os.listdir("../data/train/color")
val_dir   = os.listdir("../data/val/color")

# Training and Validation Loss History List
train_hist = []
val_hist = []
hist_cnt = 0

for i in range(EPOCHS):
    # Defining the list of this epoch loss history
    epoch_train_loss = []
    epoch_val_loss   = []

    # --- Optimize Using Training Set ---
    for data in train_dir:
        train_color = np.load("../data/train/color/"+data)
        train_depth = np.load("../data/train/depth/"+data)
        train_joy   = np.load("../data/train/joy/"+data)
        # compose color and depth
        train_depth = np.reshape(train_depth,(128,1,84,84))
        train_rgbd  = np.append(train_color,train_depth,axis=1)
        train_rgbd  = np.reshape(train_rgbd,(128,84,84,4))

        hist = model.fit(train_rgbd, train_joy,batch_size=128,verbose=1,epochs=1,validation_split=0.0)
        epoch_train_loss.append(hist.history["loss"][0])
        
    # --- Evaluate Using Load Cross-Validation Set ---
    for data in val_dir:
        val_color = np.load("../data/val/color/"+data)
        val_depth = np.load("../data/val/depth/"+data)
        val_joy   = np.load("../data/val/joy/"+data)
        print "aaa is ",str(val_joy.shape)
        # compose color and depth
        val_depth = np.reshape(val_depth,(128,1,84,84))
        val_rgbd  = np.append(val_color,val_depth,axis=1)
        val_rgbd  = np.reshape(val_rgbd,(128,84,84,4))
        # append to all rgbd and joy data
        val_loss = model.evaluate(val_rgbd,val_joy,batch_size=128)
        epoch_val_loss.append(val_loss)

    train_loss_avg = np.average(epoch_train_loss)
    train_hist.append(train_loss_avg)
    val_loss_avg   = np.average(epoch_val_loss)
    val_hist.append(val_loss_avg)
    hist_cnt += 1
    print str(hist_cnt)+"th Epoch"

# --- Display Trainig Loss ---
plt.plot(range(1,hist_cnt+1),train_hist,label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()

# --- Display Validation Loss ---
plt.plot(range(1,hist_cnt+1),val_hist,label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()

# --- Save Weight ---
time_now = str(time.time())
model.save_weights("../weights/"+time_now+".h5")
print "Saved weight"
