#!/usr/bin/env python

# General
import numpy as np
import random
import matplotlib.pyplot as plt
import os

VAL_DIR   = os.listdir("../data/train/color")

for data in VAL_DIR:
    val_color = np.load("../data/train/color/"+data)
    val_depth = np.load("../data/train/depth/"+data)
    val_joy   = np.load("../data/train/joy/"+data)
    for i in range(0,512):
        plt.imshow(val_color[i])
        plt.pause(0.01)
    print str(data)+" is finished."

