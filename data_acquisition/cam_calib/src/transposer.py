#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Pose, PoseStamped, Vector3, Quaternion, Point
import tf
import sys
import numpy as np

parent_frame = '/world'
target_frame = '/world_vive'


def transform(msg):
    position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]).reshape(3, 1)
    position = R * position + t
    msg.pose.position = Point(position[0], position[1], position[2])
    msg.header.frame_id = parent_frame
    posePublisher.publish(msg)
    # rate.sleep()


if __name__ == "__main__":
    rospy.init_node('transposer')
    tl = tf.TransformListener()
    rate = rospy.Rate(3)
    try:
        # now = rospy.Time.now()
        # tl.waitForTransform(target_frame, parent_frame, now, rospy.Duration(1))
        # (t, R) = tl.lookupTransform(target_frame, parent_frame, now)
        R = [0.70717, 0, 0, 0.70710]
        t = np.matrix([0, 0, 2]).reshape(3, 1)
        R = np.matrix(tf.transformations.quaternion_matrix(R)[:3, :3])
        print t, R
    except Exception, e:
        print e
        sys.exit(1)

    posePublisher = rospy.Publisher('/end_point_transformed', PoseStamped, queue_size=1)
    # poseListener = rospy.Subscriber('/end_point', PoseStamped, transform)
    # rospy.spin()

    while not rospy.is_shutdown():
        try:
            now = rospy.Time.now()
            msg = rospy.wait_for_message('/end_point', PoseStamped, 10)
            transform(msg)
            rate.sleep()
        except Exception, e:
            print e

