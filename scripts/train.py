#!/usr/bin/env python

# General
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# Keras
import keras
from keras import layers,models,losses
from keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Reshape
from keras.layers.normalization import BatchNormalization

# Constant Definition
EPOCHS = 50

# Initial Setting
model = modelodels.Sequential()

# --- Model Description ---
# Conv1 84 -> 40
model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu', input_shape=(84,84,4)))
model.add(BatchNormalization())
BatchNormalization
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
# Dence2 -> 25
#model.add(Dense(30))
model.add(Dense(2))

# --- load Weight ---
WEIGHT_NAME = "1579001280.0_6.h5"
model.load_weights("../weights/"+WEIGHT_NAME)


# --- Optimize Manner ---
model.compile(optimizer='adam',
              loss=losses.mean_squared_error,
              metrics=['accuracy'])
keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0)


# --- Defining Load Directory ---
TRAIN_DIR = os.listdir("../data/train/color")
VAL_DIR   = os.listdir("../data/val/color")

# --- Defining Training and Validation Loss History List ---
train_hist = []
val_hist = []
hist_cnt = 0
# each epoch loss
epoch_train_loss = []
epoch_val_loss   = []
# file name of train and validation
train_data_list = []
val_data_list = []

# --- Evaluate before training ---
# Defining the list of this epoch loss history

for data in TRAIN_DIR:
    train_data_list.append(data)
random.shuffle(train_data_list)

for data in train_data_list:
    train_color = np.load("../data/train/color/"+data)
    train_depth = np.load("../data/train/depth/"+data)
    train_joy   = np.load("../data/train/joy/"+data)
    train_joy[:,1] = train_joy[:,1]/3
    train_joy[:,0] = np.zeros(512)
    # use for rgbd learning
    train_rgbd  = np.append(train_color,train_depth,axis=3)

    train_loss = model.test_on_batch(train_rgbd,train_joy)
    epoch_train_loss.append(train_loss[0])
    
# --- Evaluate Using Load Cross-Validation Set ---
for data in VAL_DIR:
    val_data_list.append(data)
random.shuffle(val_data_list)

for data in val_data_list:
    val_color = np.load("../data/val/color/"+data)
    val_depth = np.load("../data/val/depth/"+data)
    val_joy   = np.load("../data/val/joy/"+data)
    val_joy[:,1] = val_joy[:,1]/3
    val_joy[:,0] = np.zeros(512)
    val_rgbd  = np.append(val_color,val_depth,axis=3)

    val_loss = model.test_on_batch(val_rgbd,val_joy)
    epoch_val_loss.append(val_loss[0])

train_loss_avg = np.average(epoch_train_loss)
train_hist.append(train_loss_avg)
val_loss_avg   = np.average(epoch_val_loss)
val_hist.append(val_loss_avg)

for i in range(EPOCHS):

    # --- Optimize Using Training Set ---
    for data in TRAIN_DIR:
        train_data_list.append(data)
    random.shuffle(train_data_list)

    for data in train_data_list:
        #print data

        train_color = np.load("../data/train/color/"+data)
        train_depth = np.load("../data/train/depth/"+data)
        train_joy   = np.load("../data/train/joy/"+data)
        train_joy[:,1] = train_joy[:,1]/3
        train_joy[:,0] = np.zeros(512)

        # compose color and depth
        train_rgbd  = np.append(train_color,train_depth,axis=3)

        #hist = model.fit(train_depth, train_joy,batch_size=512,verbose=0,epochs=1,validation_split=0.0,shuffle=False)
        train_loss = model.train_on_batch(train_rgbd,train_joy)
        epoch_train_loss.append(train_loss[0])
        
    # --- Evaluate Using Load Cross-Validation Set ---
    for data in VAL_DIR:
        val_data_list.append(data)
    random.shuffle(val_data_list)

    for data in val_data_list:
        val_color = np.load("../data/val/color/"+data)
        val_depth = np.load("../data/val/depth/"+data)
        val_joy   = np.load("../data/val/joy/"+data)
        val_joy[:,1] = val_joy[:,1]/3
        val_joy[:,0] = np.zeros(512)
        # compose color and depth image (128,84,84,4)
        val_rgbd  = np.append(val_color,val_depth,axis=3)
        # append to all rgbd and joy data
         
        #val_joy[:,0] = np.zeros(512)

        val_loss = model.test_on_batch(val_rgbd,val_joy)
        epoch_val_loss.append(val_loss[0])

    train_loss_avg = np.average(epoch_train_loss)
    train_hist.append(train_loss_avg)
    val_loss_avg   = np.average(epoch_val_loss)
    val_hist.append(val_loss_avg)
    hist_cnt += 1
    print str(hist_cnt)+"th Epoch"

    time_now = str(time.time())
    model.save_weights("../weights/"+time_now+"_"+str(i)+".h5")
    print "Saved weight"

# --- Display Trainig and Validation Loss ---
plt.plot(range(0,hist_cnt+1),train_hist,label="Training Loss")
plt.plot(range(0,hist_cnt+1),val_hist,label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.show()
# --- Save Loss CSV ---
np.savetxt("train_hist.csv",train_hist,delimiter=',')
np.savetxt("val_hist.csv",val_hist,delimiter=',')

# --- Save Weight ---
time_now = str(time.time())
model.save_weights("../weights/last"+time_now+".h5")
print "Saved weight"
