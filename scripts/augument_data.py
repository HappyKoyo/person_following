#!/usr/bin/env python

# General
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ROS
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# Const definision
DATA_LENGTH = 128
TARGET_DIR = "../data/train/"
TARGET_FILES = os.listdir(TARGET_DIR+"color")

for data in TARGET_FILES:
    target_color = np.load(TARGET_DIR+"color/"+data)
    target_depth = np.load(TARGET_DIR+"depth/"+data)
    target_joy   = np.load(TARGET_DIR+"joy/"+data)
    
    print target_color.dtype
    # --- generate left-right Miller data ---
    mirrored_color = target_color[:,:,::-1,:]
    mirrored_depth = target_depth[:,:,::-1,:]
    mirrored_joy   = np.dot(target_joy,np.array([[1,0],[0,-1]]))

    np.save(TARGET_DIR+"color/mirrored"+data,mirrored_color)
    np.save(TARGET_DIR+"depth/mirrored"+data,mirrored_depth)
    np.save(TARGET_DIR+"joy/mirrored"+data,mirrored_joy)
    print "mirrored data is saved"

    # visualize
    mirrored_depth = mirrored_depth.reshape(128,84,84)
    #plt.imshow(mirrored_color[0])
    #plt.pause(2)
    #plt.imshow(mirrored_depth[0],cmap="gray")
    #plt.pause(2)
    print mirrored_joy[16],target_joy[16]

print "fin"
