#!/usr/bin/env python

# General
import numpy as np
import random
import matplotlib.pyplot as plt

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

WEIGHT_NAME = "1576921462.1.h5"

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
        return resized_rgbd_img

    def main(self):
        r = rospy.Rate(10) # main loop Hz

        # --- Model Description ---
        model = models.Sequential()
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
        model.load_weights("../weights/"+WEIGHT_NAME)

        while not rospy.is_shutdown():
            r.sleep()
            # get and append data
            rgbd_input = self.getRGBD()
            vel = model.predict(rgbd_input)
            cmd_vel = Twist()
            cmd_vel.linear.x = vel[0][0]
            cmd_vel.angular.z = vel[0][1]
            print cmd_vel
            self.joy_pub.publish(cmd_vel)

if __name__ == '__main__':
    rospy.init_node('person_following_node',anonymous=True)
    person_follow = PersonFollow()
    person_follow.main()

