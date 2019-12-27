#!/usr/bin/env python

# General
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# ROS
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# Keras
import keras
from keras import layers,models,losses
from keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape, LSTM
from keras.layers.normalization import BatchNormalization

WEIGHT_NAME = "1577433216.4.h5"

class PersonFollow:
    def __init__(self):
        self.color_image_sub = rospy.Subscriber('/camera/color/image_raw',Image,self.ColorImageCB)
        self.depth_image_sub = rospy.Subscriber('/camera/depth/image_rect_raw',Image,self.DepthImageCB)
        self.joy_pub = rospy.Publisher('/cmd_vel_mux/input/teleop',Twist,queue_size=1)

        self.color_img = ()
        self.depth_img = ()
        self.joy_input = {"x":0, "theta":0}
        self.bridge = CvBridge()
        
    def ColorImageCB(self,msg):
        try:
            self.color_img = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as error_msg:
            print(error_msg)

    def DepthImageCB(self,msg):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as error_msg:
            print(error_msg)
    
    def getRGBD(self):
        # convert depth image to (float64, 1*128*128)
        resized_depth_img = cv2.resize(self.depth_img,dsize=(84,84))
        resized_depth_img = resized_depth_img.reshape(1,84,84,1)

        # reshape the color image, and compose the depth and the color image
        resized_color_img = cv2.resize(self.color_img,dsize=(84,84))
        resized_color_img = resized_color_img.reshape(1,84,84,3)
        
        # compose color and depth image (1,84,84,4)
        resized_rgbd_img  = np.append(resized_color_img,resized_depth_img,axis=3)
        print resized_rgbd_img
        return resized_depth_img#resized_rgbd_img

    def main(self):
        r = rospy.Rate(10) # main loop Hz

        # --- Model Description ---
        model = models.Sequential()
        # Conv1 84 -> 40
        model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu', input_shape=(84,84,1)))
        model.add(BatchNormalization())
        # Conv2 40 -> 18
        model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu'))
        # Conv3 18 -> 7
        model.add(Conv2D(64, kernel_size=5, strides=(2,2), activation='relu'))
        # Flatten 7*7*64 -> 3136
        model.add(Flatten())
        #model.add(Dense(128,activation="relu"))
        #model.add(Dropout(0.3))
        model.add(Dense(64,activation="relu"))
        model.add(Dropout(0.3))
        # LSTM
        model.add(Reshape((1,64)))
        model.add(LSTM(64))
        #model.add(Reshape((1,64)))
        #model.add(LSTM(64))
        # Dence2 -> 2
        #model.add(Dense(30))
        model.add(Dense(2))
        model.load_weights("../weights/"+WEIGHT_NAME)

        velocities = []
        val_data_list = []
        val_joy = 0
        VAL_DIR   = os.listdir("../data/train/color")
        for data in VAL_DIR:
            val_data_list.append(data)
        random.shuffle(val_data_list)

        for data in val_data_list:
            val_depth = np.load("../data/train/depth/"+data)
            val_joy   = np.load("../data/train/joy/"+data)
            print val_depth.shape
            val_depth = val_depth.reshape(512,84,84,1)

            for i in range(0,512):
                depth = val_depth[i]
                depth = depth.reshape(1,84,84,1)
                vel = model.predict(depth)
                velocities.append(vel)
            break
        
        velocities = np.array(velocities)
        #velocities = velocities.reshape(2,512)
        #val_joy = val_joy.reshape(2,512)
        print val_joy[0]
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

if __name__ == '__main__':
    rospy.init_node('person_following_node',anonymous=True)
    person_follow = PersonFollow()
    person_follow.main()

