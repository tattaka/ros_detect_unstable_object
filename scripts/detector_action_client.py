#! /usr/bin/env python

import cv2
import os
import time
import argparse

import actionlib
import cv_bridge
import numpy as np
import rospy
from sensor_msgs.msg import Image
from detect_unstable_object.msg import DetectUnstableObjectAction,DetectUnstableObjectGoal, DetectUnstableObjectResult
from importlib import import_module

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path')
    args = parser.parse_args()
    bridge = cv_bridge.CvBridge()
    rospy.init_node('detect_unstable_object_client')
    client = actionlib.SimpleActionClient(
        'detect_unstable_object/action_server', DetectUnstableObjectAction)
    client.wait_for_server()

    goal = DetectUnstableObjectGoal()
    image = cv2.imread(args.img_path)
    goal.image = bridge.cv2_to_imgmsg(image)
    goal.id = int(time.time())
    client.send_goal(goal)
    client.wait_for_result(rospy.Duration.from_sec(5.0))
