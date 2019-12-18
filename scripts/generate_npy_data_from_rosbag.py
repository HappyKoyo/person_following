#!/usr/bin/env python

# General
import numpy as np
import random

# ROS
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# Const definision
DATA_LENGTH = 128
DIR_TYPE = "train"# train, val or test. 

class GenTrainData:
    def __init__(self):
        self.color_image_sub = rospy.Subscriber('/camera/color/image_raw',Image,self.ColorImageCB)
        self.depth_image_sub = rospy.Subscriber('/camera/depth/image_rect_raw',Image,self.DepthImageCB)
        self.joy_input_sub   = rospy.Subscriber('/teleop_velocity_smoother/raw_cmd_vel',Twist,self.JoyInputCB)

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
    
    def JoyInputCB(self,msg):
        self.joy_input["x"]     = msg.linear.x
        self.joy_input["theta"] = msg.angular.z

    def getTrainDepth(self):
        # convert image to (float64, 1*84*84)
        resized_depth_img = cv2.resize(self.depth_img,dsize=(84,84))
        resized_depth_img = resized_depth_img.astype(np.float64)
        resized_depth_img = resized_depth_img.reshape(1,84,84)
        # scaling depth from 0 to 1
        for h_i in range(84):
            for w_i in range(84):
                if resized_depth_img[0][h_i][w_i] < 1 or 3000 < resized_depth_img[0][h_i][w_i]:
                    resized_depth_img[0][h_i][w_i] = 0
                else:
                    resized_depth_img[0][h_i][w_i] = 1-float(resized_depth_img[0][h_i][w_i]-30)/2980
        return resized_depth_img

    def getTrainColor(self):
        # convert image to (float64, 3*84*84)
        resized_color_img = cv2.resize(self.color_img,dsize=(84,84))
        resized_color_img = resized_color_img.astype(np.float64)
        resized_color_img = resized_color_img.reshape(1,3,84,84)
        return resized_color_img
    
    def getTrainJoy(self):
        joy_data = np.zeros(2).reshape(1,2)
        joy_data[0][0] = self.joy_input["x"]
        joy_data[0][1] = self.joy_input["theta"]
        return joy_data

    def main(self):
        r = rospy.Rate(10) # main loop Hz
        img_num = 0
        # create first black image
        all_color = np.zeros(3*84*84).reshape(1,3,84,84)
        all_depth = np.zeros(1*84*84).reshape(1,84,84)
        all_joy   = np.zeros(2).reshape(1,2)
        while not rospy.is_shutdown() and img_num < DATA_LENGTH:
            r.sleep()
            img_num = img_num + 1
            # getting 1 data
            color_data = self.getTrainColor()
            depth_data = self.getTrainDepth()
            joy_data   = self.getTrainJoy()
            # appending a frame
            all_color = np.append(all_color,color_data,axis=0)
            all_depth = np.append(all_depth,depth_data,axis=0)
            all_joy   = np.append(all_joy,joy_data,axis=0)
            print img_num
        # delete first black image
        print all_color.shape
        train_color = np.delete(all_color,0,axis=0)
        print train_color.shape
        train_depth = np.delete(all_depth,0,axis=0)
        train_joy   = np.delete(all_joy,0,axis=0)
        # save each data
        time = str(rospy.get_time())
        np.save('../data/'+DIR_TYPE+'/npy/color/'+time+'.npy',train_color)
        np.save('../data/'+DIR_TYPE+'/npy/depth/'+time+'.npy',train_depth)
        np.save('../data/'+DIR_TYPE+'/npy/joy/'+time+'.npy',train_joy)
        print "training data "+time+".npy is saved in /data/"+DIR_TYPE+"/train/npy/ each director."

if __name__ == '__main__':
    rospy.init_node('generate_train_data_from_rosbag',anonymous=True)
    gen_train = GenTrainData()
    gen_train.main()
