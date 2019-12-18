#!/usr/bin/env python

# ROS
import rospy

# GENERAL
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class CheckImage:
    def __init__(self):
        self.color_image_sub = rospy.Subscriber('/camera/color/image_raw',Image,self.ColorImageCB)
        self.depth_image_sub = rospy.Subscriber('/camera/depth/image_rect_raw',Image,self.DepthImageCB)
        self.resized_color_pub = rospy.Publisher('/resized/color', Image, queue_size=1)
        self.resized_depth_pub = rospy.Publisher('/resized/depth', Image, queue_size=1)

        self.bridge = CvBridge()
        
    def ColorImageCB(self,msg):
        try:
            self.color_img = self.bridge.imgmsg_to_cv2(msg)
            self.resized_image = self.color_img
            resized_image = cv2.resize(self.color_img,dsize=(128,128))
            resized_image = self.bridge.cv2_to_imgmsg(resized_image,"rgb8")
            self.resized_color_pub.publish(resized_image)
        except CvBridgeError as error_msg:
            print(error_msg)

    def DepthImageCB(self,msg):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg)
            resized_image = cv2.resize(self.depth_img,dsize=(128,128))
            resized_image = self.bridge.cv2_to_imgmsg(resized_image,"mono16")
            self.resized_depth_pub.publish(resized_image)
        except CvBridgeError as error_msg:
            print(error_msg)

if __name__ == '__main__':
    rospy.init_node('check_image',anonymous=True)
    ci = CheckImage()
    rospy.spin()
