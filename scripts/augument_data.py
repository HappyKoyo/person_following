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
TARGET_DIR = "../data/train/"
TARGET_FILES = os.listdir(TARGET_DIR+"color")

# --- generate left-right Miller data ---
for data in TARGET_FILES:
    target_color = np.load(TARGET_DIR+"color/"+data)
    target_depth = np.load(TARGET_DIR+"depth/"+data)
    target_joy   = np.load(TARGET_DIR+"joy/"+data)
    
    mirrored_color = target_color[:,:,::-1,:]
    mirrored_depth = target_depth[:,:,::-1,:]
    mirrored_joy   = np.dot(target_joy,np.array([[1,0],[0,-1]]))
    np.save(TARGET_DIR+"color/mirrored"+data,mirrored_color)
    np.save(TARGET_DIR+"depth/mirrored"+data,mirrored_depth)
    np.save(TARGET_DIR+"joy/mirrored"+data,mirrored_joy)
    print "mirrored "+data+" is saved"
print "mirrored data is saved"
    
# --- generate bright changed data (deleted because the network has BatchNormalization)---
#alpha = 0.5
#beta = 0.0
#illuminated_color = np.clip(alpha*target_color+beta,0,255).astype(np.uint8)

"""
TARGET_FILES = os.listdir(TARGET_DIR+"color")
# --- generate rgb swap data ---
for data in TARGET_FILES:
    target_color = np.load(TARGET_DIR+"color/"+data)
    target_depth = np.load(TARGET_DIR+"depth/"+data)
    target_joy   = np.load(TARGET_DIR+"joy/"+data)
    # --- generate rbg data
    rbg_color = target_color[:,:,:,[0,2,1]]
    brg_color = target_color[:,:,:,[1,0,2]]
    bgr_color = target_color[:,:,:,[1,2,0]]
    grb_color = target_color[:,:,:,[2,0,1]]
    gbr_color = target_color[:,:,:,[2,1,0]]

    np.save(TARGET_DIR+"color/rbg"+data,rbg_color)
    np.save(TARGET_DIR+"depth/rbg"+data,target_depth)
    np.save(TARGET_DIR+"joy/rbg"+data,target_joy)

    np.save(TARGET_DIR+"color/brg"+data,brg_color)
    np.save(TARGET_DIR+"depth/brg"+data,target_depth)
    np.save(TARGET_DIR+"joy/brg"+data,target_joy)

    np.save(TARGET_DIR+"color/bgr"+data,bgr_color)
    np.save(TARGET_DIR+"depth/bgr"+data,target_depth)
    np.save(TARGET_DIR+"joy/bgr"+data,target_joy)

    np.save(TARGET_DIR+"color/grb"+data,grb_color)
    np.save(TARGET_DIR+"depth/grb"+data,target_depth)
    np.save(TARGET_DIR+"joy/grb"+data,target_joy)

    np.save(TARGET_DIR+"color/gbr"+data,gbr_color)
    np.save(TARGET_DIR+"depth/gbr"+data,target_depth)
    np.save(TARGET_DIR+"joy/gbr"+data,target_joy)

    # visualize
    #illuminated_depth = mirrored_depth.reshape(128,84,84)
    #plt.imshow(mirrored_color[0])
    #plt.pause(2)
    print "rgb swap data is saved"
"""
print "fin"
