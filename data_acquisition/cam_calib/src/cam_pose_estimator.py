#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: syxtreme
"""
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import sys
import wx

class ImageViewPanel(wx.Panel):
    """ class ImageViewPanel creates a panel with an image on it, inherits wx.Panel """
    def update(self, image):
        if not hasattr(self, 'staticbmp'):
            self.staticbmp = wx.StaticBitmap(self)
            frame = self.GetParent()
            frame.SetSize((image.shape[1], image.shape[0]))
        bmp = wx.BitmapFromBuffer(image.shape[1], image.shape[0], image)
        self.staticbmp.SetBitmap(bmp)

class CameraPoseEstimator(wx.App):
    PHASE_CAM_EXT_CALIB = 1
    PHASE_CAM_POSE = 2
    PHASE_TIP_COLOR = 4
    PHASE_TIP_POSE = 8

    def OnInit(self):
        # ROS initialization
        rospy.init_node("pose_estimator")
#        self.nodeName = rospy.get_name()
        # Get the parameters
        imageTopic = rospy.get_param('~image', '')
        cinfoTopic = rospy.get_param('~cinfo', '')
        # Get camera info
        rospy.loginfo('Waiting for camera info on topic {}.'.format(cinfoTopic))
        self.cinfo = rospy.wait_for_message(cinfoTopic, CameraInfo)
        # Calculate required variables and stuff
        rospy.loginfo('Camera info recieved, calculating stuff.')
        self.mtx = np.float64(self.cinfo.K).reshape((3, 3))
        self.dist = np.float64(self.cinfo.D).reshape((1,5))
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((6*8,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
        self.axis = np.float32([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1,3)
        self.bridge = CvBridge()
        # Subscribe to the image topic
        rospy.loginfo('Subscribing to images on topic {}.'.format(imageTopic))
        self.imageSubscriber = rospy.Subscriber(imageTopic, CompressedImage, callback=self.estPose, queue_size=1)
        # Setup necessary variables
        self.currentPhase = self.PHASE_CAM_EXT_CALIB
        self.rvecsBuffer = []
        self.tvecsBuffer = []

        # >>> Setup GUI <<<
        self.frame = wx.Frame(None, title = "Calibrator", size = (256, 256))
        self.panel = ImageViewPanel(self.frame)
        toolbar = wx.ToolBar(self.frame, -1)
        self.nextButton = wx.Button(toolbar, -1, 'Next phase')
        self.frame.Bind(wx.EVT_BUTTON, self.onNextButton, source=self.nextButton)
        toolbar.AddControl(self.nextButton)
        toolbar.Realize()
        self.frame.SetToolBar(toolbar)

        self.frame.Show(True)
        return True


    def onNextButton(self, event):
        if self.currentPhase == self.PHASE_CAM_EXT_CALIB:
            self.tvecs = np.mean(self.tvecsBuffer, axis=0)
            self.rvecs = np.mean(self.rvecsBuffer, axis=0)
        self.currentPhase <<= 1

    def updateImage(self, img):
        img = cv2.resize(img, (img.shape[1] >> 1, img.shape[0] >> 1))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.panel.update(img)

    def drawAxis(self, img, imgpts):
        origin = tuple(imgpts[0].ravel())
        img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (255,0,0), 5)
        img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,255,0), 5)
        img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (0,0,255), 5)
        return img

    def estPose(self, msg):
        print msg.header
        img = self.bridge.compressed_imgmsg_to_cv2(msg)

        if self.currentPhase == self.PHASE_CAM_EXT_CALIB:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (8, 6), None, cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_FILTER_QUADS | cv2.CALIB_CB_ADAPTIVE_THRESH)

            if ret == True:
                corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), self.criteria)
                # Find the rotation and translation vectors.
                ret, rvecs, tvecs = cv2.solvePnP(self.objp, corners2, self.mtx, None)
                self.rvecsBuffer.append(rvecs)
                self.tvecsBuffer.append(tvecs)
            else:
                corners2 = corners
            img = cv2.drawChessboardCorners(img, (8, 6), corners2, ret)
        elif self.currentPhase == self.PHASE_CAM_POSE:
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(self.axis, self.rvecs, self.tvecs, self.mtx, None)
            img = self.drawAxis(img, imgpts)
        elif self.currentPhase == self.PHASE_TIP_COLOR:
            pass
        elif self.currentPhase == self.PHASE_TIP_POSE:
#            img = wx.CallAfter(self.drawAxis, img, imgpts)
            pass

        wx.CallAfter(self.updateImage, img)

def main(args):
    pe = CameraPoseEstimator()
    pe.MainLoop()
    print("Shutting down")
#   cv2.destroyAllWindows()
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))