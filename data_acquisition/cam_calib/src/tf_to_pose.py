#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: syxtreme
"""
import rospy
from geometry_msgs.msg import Pose, PoseStamped, Vector3, Quaternion
import tf
import sys


def main(args):
    rospy.init_node('tf_to_pose')
    parent_frame = rospy.get_param('~parent_frame', '/world_vive')
    target_frame = rospy.get_param('~target_frame', '/')
    target_topic = rospy.get_param('~target_topic', '')
    HZ = rospy.get_param('~hz', 60)
    if not target_topic:
        target_topic = target_frame

    rospy.loginfo('Started transmitting pose on topic "{}" from TF frame id: "{}" with respect to frame "{}"'.format(
        target_topic, target_frame, parent_frame))
    rate = rospy.Rate(HZ)

    tfListener = tf.TransformListener()
    posePublisher = rospy.Publisher(target_topic, PoseStamped, queue_size=10)

    while not rospy.is_shutdown():
        try:
            time = tfListener.getLatestCommonTime(parent_frame, target_frame)
            (trans, rot) = tfListener.lookupTransform(
                parent_frame, target_frame, time)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rate.sleep()
            continue

        pose = PoseStamped()
        pose.header.stamp = time
        pose.header.frame_id = parent_frame
        pose.pose.position = Vector3(*trans)
        pose.pose.orientation = Quaternion(*rot)
        posePublisher.publish(pose)
        rate.sleep()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
