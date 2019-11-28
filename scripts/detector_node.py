#! /usr/bin/env python

import rospy

from detector import UnstableObjectDetector

if __name__ == '__main__':
    rospy.init_node('detect_unstable_object')
    UnstableObjectDetector()
    rospy.spin()
