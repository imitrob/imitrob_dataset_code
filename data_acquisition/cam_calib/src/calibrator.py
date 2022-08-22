#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: syxtreme
"""
import cv2
import numpy as np
import rospy
import rostopic
import rospkg
from sensor_msgs.msg import CameraInfo, CompressedImage  # noqa
from geometry_msgs.msg import Pose, Vector3, Quaternion, TransformStamped, PoseStamped  # noqa
from visualization_msgs.msg import Marker, MarkerArray  # noqa
from cv_bridge import CvBridge
import os
import sys
import wx
import uuid
import tf
import datetime
import pygame as pg
import scipy as sp
from scipy.linalg import sqrtm
import scipy.odr
import pynput
import tf2_ros
import transforms3d as tf3
import open3d as o3d
import colored


print("WX Version (should be 4+): {}".format(wx.__version__))
CHECKERBOARD_SIZE = (8, 6)  # (width, height)
CHECKERBOARD_CELL_SIZE = 0.072  # in meters
WORLD_FRAME = 'board'               # World (base) frame for the camera
TRACKED_FRAME = 'world_vive'        # Name of the world (base) frame for the tracked topic
CALIB_MAX_TRANSFORM_VECTORS = 10  # how many samples to collect during extrinsic cam calibration
TRACKER_PLANE_OFFSET = 0.0  # offset between physical checkerboard and the endpoint of the calibration "spike" endpoint; e.g., when using a pitted plexiglass (for better grip of the spike and to protect the board)
MARKER_LENGTH = 0.050  # mm


def getTopicListBAndEdit(topic, parent):
    """Function generates wxPython control elements (listbox and editbox) for the specific topic
    and with the specified parent window. Specify the topics to search for -- the list box will
    contain all currently active topics of the specified type.
    When typing to the edit box, topics in the listbox are automatically filtered out (only topics
    matching the typed text are kept). The listbox will be replaced by a static text saying
    that no topics were found if no topics of the specified type are found.

    Arguments:
        topic {string | list} -- The (list of) topic(s) to search for.
        parent {wx.Window} -- Parent wx window of the created controls (see wxPython for more).

    Returns:
        wx.ListBox | wx.StaticText -- The listbox containing the topics or a static text
        wx.TextCtrl -- The editbox for manually providing the topic name or searching the list of topics
    """

    if type(topic) is str:
        topic = [topic]
    topics = []
    for t in topic:
        topics += rostopic.find_by_type(t)
    editBox = wx.TextCtrl(parent)
    if len(topics) == 0:
        return wx.StaticText(parent, -1, 'No topics of type {} found.'.format(topic)), editBox
    listBox = wx.ListBox(parent, choices=topics)
    listBox.Select(0)

    topics = np.array(topics)
    parent.Bind(wx.EVT_TEXT, lambda event: listBox.SetItems(topics[map(lambda t: editBox.GetValue().lower() in t, np.core.defchararray.lower(topics))]), source=editBox)
    parent.Bind(wx.EVT_LISTBOX, lambda event: editBox.ChangeValue(event.GetString()), source=listBox)
    return listBox, editBox


def colorValue(value, max_val, min_val):
    percentage = (value - min_val) / (max_val - min_val)
    fg_val, bg_val = 15, 0
    if percentage < 0:
        if percentage < -1:
            fg_val = 109
    elif percentage > 1:
        if percentage > 2:
            bg_val = 124
        else:
            fg_val = 124
    else:
        color_range = [195, 190, 191, 192, 186, 185, 184, 178, 172, 166, 160, 196]
        scale = int(percentage * (len(color_range)))
        fg_val = color_range[scale]

    return "{}{} {} {}".format(colored.fg(fg_val), colored.bg(bg_val) if bg_val > 0 else "", value, colored.style.RESET)


class PoseEstimator():
    SQUARE_LENGTH = CHECKERBOARD_CELL_SIZE  # mm
    BOARD_COLS, BOARD_ROWS = CHECKERBOARD_SIZE

    CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    try:
        dictionary = cv2.aruco.Dictionary_create(48, 4, 65536)
    except Exception as e:   # noqa
        dictionary = cv2.aruco.Dictionary_create(48, 4)  # TODO hack for (old?) cv2 version, fallback to API with 2 args only

    pnpFlags = cv2.SOLVEPNP_IPPE
    chessFlags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH

    def __init__(self, cameraMatrix, distCoeffs):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.resetBoardOrientation()

    def resetBoardOrientation(self):
        self.boardOrientationSet = False
        self.__objp = np.zeros((self.BOARD_COLS * self.BOARD_ROWS, 3), np.float32)
        self.__objp[:, :2] = np.mgrid[0:self.BOARD_COLS, 0:self.BOARD_ROWS].T.reshape(-1, 2) * self.SQUARE_LENGTH
        self.__gridpoints = np.zeros((CHECKERBOARD_SIZE[1] * CHECKERBOARD_SIZE[0], 3), np.float32)
        self.__gridpoints[:, :2] = np.mgrid[0:self.BOARD_COLS, 0:self.BOARD_ROWS].T.reshape(-1, 2)

    def estimateBoardPose(self, image, init_rvec=None, init_tvec=None):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        b_rvecs, b_tvecs = None, None
        origin_imgCoord = None

        markerCorners, markerIds, rejectedPts = cv2.aruco.detectMarkers(image, self.dictionary, cameraMatrix=self.cameraMatrix, distCoeff=self.distCoeffs)
        # image = cv2.aruco.drawDetectedMarkers(image, markerCorners, markerIds)
        if markerIds is not None:
            if len(markerIds) >= 2:
                markerCorners = np.squeeze(markerCorners)
            else:
                rospy.logwarn_once("Only one marker detected, origin pose might be incorrect!")

            origin_imgCoord = np.reshape(markerCorners, (-1, 2)).mean(0)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, MARKER_LENGTH, self.cameraMatrix, self.distCoeffs)
            # for rvec, tvec in zip(rvecs, tvecs):
            #     image = cv2.aruco.drawAxis(image, self.cameraMatrix, self.distCoeffs, rvec, tvec, 0.05)

        else:
            rospy.logerr_once("Cannot detect markers, calibration chessboard might be detected with a wrong orientation!")

        ret, corners = cv2.findChessboardCorners(image, (self.BOARD_COLS, self.BOARD_ROWS), flags=self.chessFlags)

        if ret:
            corners2 = cv2.cornerSubPix(image, corners, (15, 15), (-1, -1), self.CRITERIA)
            # Find the rotation and translation vectors.
            if not self.boardOrientationSet:
                if not(origin_imgCoord is None or np.isclose(corners2[0, :], origin_imgCoord, atol=20).all()):
                    self.__objp = np.flipud(self.__objp)
                    self.__gridpoints = np.flipud(self.__gridpoints)
                self.boardOrientationSet = True
            else:
                if not(origin_imgCoord is None or np.isclose(corners2[0, :], origin_imgCoord, atol=20).all()):
                    raise Exception("Markers indicate different orientation of the board than what was used in previous calibrations!")

            # rospy.loginfo(corners2.shape)
            ret, b_rvecs, b_tvecs, inliers = cv2.solvePnPRansac(self.objp, corners2, self.cameraMatrix, self.distCoeffs,
                                                                rvec=init_rvec, tvec=init_tvec, useExtrinsicGuess=not(
                                                                    init_rvec is None and init_tvec is None),
                                                                confidence=0.999, iterationsCount=500, reprojectionError=2, flags=self.pnpFlags)
            corners = corners2
            # rospy.loginfo(np.array2string(inliers))

        return b_rvecs is not None, b_rvecs, b_tvecs, corners

    @property
    def objp(self):
        return self.__objp

    @property
    def gridpoints(self):
        return self.__gridpoints


class ImageViewPanel(wx.Panel):
    """ class ImageViewPanel creates a panel with an image on it, inherits wx.Panel """
    def __init__(self, parent, sampleCallback=None):
        self.width = 1024
        self.height = 768
        super(ImageViewPanel, self).__init__(parent, id=-1, size=wx.Size(self.width, self.height))
        self.bitmap = None
        self.Bind(wx.EVT_PAINT, self.onPaint)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)

        self.Bind(wx.EVT_LEFT_DCLICK, self.onClick)
        self.Bind(wx.EVT_LEFT_DOWN, self.onMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.onMouseUp)
        self.Bind(wx.EVT_MOTION, self.onMouseMove)

        self.mouseIsDown = False
        self.trackingStart = None
        self.trackingStop = None
        self.selection = None
        self.sampleCallback = sampleCallback
        self.brush = wx.Brush(wx.Colour(200, 200, 255, 128), wx.BRUSHSTYLE_FDIAGONAL_HATCH)

    def onClick(self, event):
        if not self.sampleCallback:
            return
        pos = event.GetPosition()
        if self.bitmap:
            rect = wx.Rect(pos.x, pos.y, 10, 10)
            sample = np.asarray(self.bitmap.GetSubBitmap(rect).ConvertToImage().GetDataBuffer(), dtype=np.uint8).reshape((-1, 1, 3))
            wx.CallAfter(self.sendSample, sample, rect)

    def onMouseDown(self, event):
        if not self.sampleCallback:
            return
        self.mouseIsDown = True
        self.trackingStart = event.GetPosition()

    def onMouseUp(self, event):
        if not (self.sampleCallback and self.trackingStart and self.trackingStop):
            return
        self.mouseIsDown = False
        a, b = (self.trackingStart, self.trackingStop) if self.trackingStart < self.trackingStop else (self.trackingStop, self.trackingStart)
        rect = wx.Rect(a, b)
        size = rect.GetSize()
        # Send the samples only if the selection area is bigger than 64 pixels
        if size[0] * size[1] < 16:
            self.selection = None
        else:
            self.selection = rect
            self.Refresh()

    def onMouseMove(self, event):
        if not (self.sampleCallback and self.mouseIsDown):
            return
        self.trackingStop = event.GetPosition()
        a, b = (self.trackingStart, self.trackingStop) if self.trackingStart < self.trackingStop else (self.trackingStop, self.trackingStart)
        self.selection = wx.Rect(a, b)
        self.Refresh()

    def update(self, image):
        if image.shape[2] == 4:
            alpha = image[:, :, 3].astype(np.uint8)
            image = image[:, :, :3].astype(np.uint8)
            image = wx.Image(image.shape[1], image.shape[0], image.tostring(), alpha.tostring())
        else:
            # print(image.astype(np.uint8).shape)
            # print(dir(wx.Image))
            # print(dirwx.Image.__module__)
            image = wx.Image(image.shape[1], image.shape[0], image.astype(np.uint8).ravel().tostring())
            # image = wx.Image(name="RGB imae", width=image.shape[1], height=image.shape[0], data=image.astype(np.uint8).tostring())
            # image = wx.Image(image.shape[1], image.shape[0], image.astype(np.uint8).tostring())
        self.imgWidth, self.imgHeight = image.GetSize()
        if self.imgWidth > self.width or self.imgHeight > self.height:
            self.ratio = float(self.height) / self.imgHeight
            self.bitmap = wx.Bitmap(image.Scale(self.imgWidth * self.ratio, self.height))
        else:
            self.bitmap = wx.Bitmap(image)
        self.Refresh()

    def backmap(self, x, y):
        return x * self.ratio, y * self.ratio

    def sendSample(self, sample, rect):
        invRatio = 1 / self.ratio
        ogrid = np.ogrid[int(np.round(rect.Top * invRatio)):int(np.round(rect.Bottom * invRatio)), int(np.round(rect.Left * invRatio)):int(np.round(rect.Right * invRatio)), 0:3]
        self.sampleCallback(sample, ogrid)

    def onPaint(self, event):
        if self.bitmap:
            # img_w, img_h = self.bitmap.GetSize()
            margin = 0
            self.SetSize(wx.Size(self.imgWidth + margin * 2, self.imgHeight + margin * 2))
            dc = wx.AutoBufferedPaintDC(self)
            dc.Clear()
            dc.DrawBitmap(self.bitmap, margin, margin, useMask=True)
            if self.sampleCallback and self.selection:
                dc.SetBrush(self.brush)
                dc.DrawRectangle(self.selection)
                sample = np.asarray(self.bitmap.GetSubBitmap(self.selection).ConvertToImage().GetDataBuffer(),
                                    dtype=np.uint8).reshape((-1, 1, 3))
                # If mouse is up and something is still selected, send the sample
                if not self.mouseIsDown:
                    # opcvsamp = np.asarray(self.bitmap.GetSubBitmap(self.selection).ConvertToImage().GetDataBuffer()).reshape((self.selection.Height, self.selection.Width, 3))
                    # print(self.selection.GetSize())
                    # print(opcvsamp.shape)
                    # wx.CallAfter(cv2.imshow, 'sample', opcvsamp)
                    wx.CallAfter(self.sendSample, sample, self.selection)
                    self.selection = None


class ColorTrackingParams():
    def __init__(self, **kwargs):
        self.addParams(**kwargs)

    def addParams(self, **kwargs):
        # TODO: Add paramter checking
        for (key, value) in kwargs.iteritems():
            setattr(self, key, value)


class TrackerFrame(wx.Frame):
    PHASE_WAITING_POSE_TOPIC = 1
    PHASE_WAITING_IMAGE_TOPIC = 2
    PHASE_IMAGE_CALIBRATION = 4
    PHASE_OBJECT_MARKING = 8
    PHASE_OBJECT_DETECTION = 16
    PHASE_POSE_CALIBRATION = 32
    PHASE_DONE = 64
    PHASE_AWAITING_USER_INPUT = 128

    PHASE_TRACKING = 256
    PHASE_POINT_GATHERING = 512

    HSV_VALUE_BLUR = 0.2
    LAB_VALUE_BLUR = 0.1
    CONTOUR_AREA_SIZE_THRESHOLD = 32

    GAUGE_RANGE = 1000

    UPDATE_INTERVAL = 1000

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # gridpoints = np.zeros((CHECKERBOARD_SIZE[1] * CHECKERBOARD_SIZE[0], 3), np.float32)
    # gridpoints[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)

    # objp = np.zeros((CHECKERBOARD_SIZE[1] * CHECKERBOARD_SIZE[0], 3), np.float64)
    # objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0] * CHECKERBOARD_CELL_SIZE:CHECKERBOARD_CELL_SIZE, 0:CHECKERBOARD_SIZE[1] * CHECKERBOARD_CELL_SIZE:CHECKERBOARD_CELL_SIZE].T.reshape(-1, 2)
    area = np.float32([[0, 0, 0], [(CHECKERBOARD_SIZE[0] - 1) * CHECKERBOARD_CELL_SIZE, 0, 0], [(CHECKERBOARD_SIZE[0] - 1) * CHECKERBOARD_CELL_SIZE, (CHECKERBOARD_SIZE[1] - 1) * CHECKERBOARD_CELL_SIZE, 0], [0, (CHECKERBOARD_SIZE[1] - 1) * CHECKERBOARD_CELL_SIZE, 0], [0, 0, 3 * CHECKERBOARD_CELL_SIZE]]).reshape(-1, 3)

    def __init__(self, trackedTopic, camera):
        super(TrackerFrame, self).__init__(None, title='Tip Tracker', size=wx.Size(1024, 768))
        self.trackedTopic = trackedTopic
        rospy.loginfo('Selected topic for tracking: {}'.format(trackedTopic))
        rospy.loginfo('Image topic: {}, camera info topic: {}'.format(camera.imageTopic, camera.cinfoTopic))

        rospack = rospkg.RosPack()
        self.packagePath = rospack.get_path('cam_calib')
        self.camera = camera
        self.tfBroadcaster = tf.TransformBroadcaster()
        self.topicFrameID = None

        self.rvecsBuffer = []
        self.tvecsBuffer = []
        self.topicFreshnessCounter = rospy.Time.now()
        self.colorTrackingParams = None
        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.contourAreaSizeThreshold = self.CONTOUR_AREA_SIZE_THRESHOLD
        self.gpPositiveSamp = []
        self.gpNegativeSamp = []
        self.isCalibrated = False

        print('initializing PyGame')
        pg.init()
        print('initializing PG Mixer')
        pg.mixer.init()
        print('trying to import sounds')
        self.soundCoin = pg.mixer.Sound(os.path.join(self.packagePath, 'src', 'media', 'coin.wav'))
        self.soundWin = pg.mixer.Sound(os.path.join(self.packagePath, 'src', 'media', 'mario_victory.wav'))
        self.soundWin.set_volume(0.5)
        self.soundHorn = pg.mixer.Sound(os.path.join(self.packagePath, 'src', 'media', 'inception.ogg'))
        self.soundHorn.set_volume(0.5)
        self.soundMight = pg.mixer.Sound(os.path.join(self.packagePath, 'src', 'media', 'test_your_might.ogg'))
        self.soundBeep = pg.mixer.Sound(os.path.join(self.packagePath, 'src', 'media', 'beep.wav'))

        self.bridge = CvBridge()
        self.phase = self.PHASE_WAITING_IMAGE_TOPIC | self.PHASE_WAITING_POSE_TOPIC

        # Toolbar
        toolbar = wx.ToolBar(self, -1)
        toolbar.SetToolSeparation(20)
        self.buttonCollect = wx.Button(toolbar, -1, 'Collect points', name="buttonCollect")
        self.Bind(wx.EVT_BUTTON, self.onCollectButton, source=self.buttonCollect)
        self.buttonCollect.Disable()
        toolbar.AddControl(self.buttonCollect)
        toolbar.AddStretchableSpace()
        self.buttonIFeed = wx.Button(toolbar, -1, 'Deactivate camera', name="buttonImageFeed")
        self.Bind(wx.EVT_BUTTON, self.onImageFeedButton, source=self.buttonIFeed)
        toolbar.AddControl(self.buttonIFeed)
        toolbar.AddStretchableSpace()
        self.buttonLoad = wx.Button(toolbar, -1, 'Load points', name="buttonLoad")
        self.Bind(wx.EVT_BUTTON, self.onLoadButton, source=self.buttonLoad)
        self.buttonLoad.Disable()
        toolbar.AddControl(self.buttonLoad)
        toolbar.AddStretchableSpace()
        button = wx.Button(toolbar, -1, 'Save points', name="buttonSave")
        self.Bind(wx.EVT_BUTTON, self.onSaveButton, source=button)
        toolbar.AddControl(button)
        self.buttonCalib = wx.Button(toolbar, -1, 'Calibrate', name="buttonCalib")
        self.Bind(wx.EVT_BUTTON, self.onCalibButton, source=self.buttonCalib)
        toolbar.AddControl(self.buttonCalib)
        toolbar.Realize()
        self.SetToolBar(toolbar)
        self.vbox.Add(toolbar, flag=wx.EXPAND)

        # Info text
        self.infoText = wx.TextCtrl(self, -1, value='Waiting for messages from the tracked topic', style=wx.TE_READONLY | wx.BORDER_NONE)
        font = wx.Font(wx.FontInfo(42).Bold())
        tattr = wx.TextAttr()
        tattr.SetFont(font)
        self.infoText.SetDefaultStyle(tattr)
        self.vbox.Add(self.infoText, flag=wx.EXPAND)

        self.gauge = wx.Gauge(self, range=self.GAUGE_RANGE, name="gaugeCalib")
        self.vbox.Add(self.gauge, flag=wx.EXPAND)

        self.box = wx.FlexGridSizer(0, 1, 10, 10)

        self.checkTrackedTopic = wx.CheckBox(self, -1, label='Tracked topic received')
        self.checkTrackedTopic.Bind(wx.EVT_LEFT_DOWN, lambda x: None)
        self.checkTrackedTopic.Bind(wx.EVT_LEFT_UP, lambda x: None)
        self.box.Add(self.checkTrackedTopic, flag=wx.EXPAND)

        self.vbox.Add(self.box, flag=wx.EXPAND)

        # Image view
#        self.imageView = ImageViewPanel(self, sampleCallback=self.onSampleColorReceived)
        self.imageView = ImageViewPanel(self)
        self.vbox.Add(self.imageView, flag=wx.EXPAND)

        self.currentPoint = 0
        self.points = []
        self.orientations = []

        self.SetSizerAndFit(self.vbox)
        self.Show(True)
        self.SetFocus()
        self.Bind(wx.EVT_CLOSE, self.onClose, source=self)

        self.posePoints = np.empty(CHECKERBOARD_SIZE, dtype=np.object)
        self.poseOrientations = np.empty(CHECKERBOARD_SIZE, dtype=np.object)

        self.markerIDs = np.zeros(CHECKERBOARD_SIZE)
        self.markerColors = np.zeros(CHECKERBOARD_SIZE + (3,))
        for ind in np.ndindex(CHECKERBOARD_SIZE):
            self.markerIDs[ind] = int(str(ind[0]) + str(ind[1])) + 100
            self.markerColors[ind] = np.random.rand(3)
        self.miniPointMarkers = []
        self.gridMarkers = []
        self.markerPublisher = rospy.Publisher('~/calibration_markers', Marker, queue_size=100)

        self.trackedPoint = None
        self.tfListener = tf.TransformListener()
        self.posePublisher = rospy.Publisher('/end_point', PoseStamped, queue_size=10)

        toolbar.Bind(wx.EVT_KEY_DOWN, self.onKeyDownOverButton)
        self.Bind(wx.EVT_KEY_DOWN, self.onKeyDownOverButton)
        for child in self.GetChildren():
            child.Bind(wx.EVT_KEY_DOWN, self.onKeyDownOverButton)
        for child in toolbar.GetChildren():
            child.Bind(wx.EVT_KEY_DOWN, self.onKeyDownOverButton)
        self.keyDaemon = pynput.keyboard.Listener(on_press=self.onKeyDown)
        self.keyDaemon.daemon = True
        self.keyDaemon.start()

        self.trackingSubscriber = rospy.Subscriber(trackedTopic, rostopic.get_topic_class(trackedTopic)[0], self.trackedTopicCallback)
        self.imageSubscriber = rospy.Subscriber(camera.imageTopic, rostopic.get_topic_class(camera.imageTopic)[0], self.updateImage)

    transmitter = tf.TransformBroadcaster()
    poser = rospy.Publisher('/test_pose', PoseStamped, queue_size=10)

    @staticmethod
    def Rt_end2cam(R_controller2world, t_controller2world):

        R_controller2world = tf.transformations.quaternion_matrix(R_controller2world)
        t_controller2world = tf.transformations.translation_matrix(t_controller2world)

        # const
        t_end2controller = np.array([0.00481989, 0.04118873, 0.21428937, 1]).reshape(4, 1)
        R_0 = tf.transformations.euler_matrix(0, -np.pi / 2, 0)
        t_board2world = tf.transformations.translation_matrix([-0.236220294307, -1.2843890696, -3.01909357602])
        R_board2world = tf.transformations.quaternion_matrix([0.506393272061, -0.488596625235, 0.511542026753, 0.493116565009])
        t_cam2board = tf.transformations.translation_matrix([0.28536, 0.63506, -0.98828])
        R_cam2board = tf.transformations.quaternion_matrix([0.27406, -0.016329, -0.0074531, 0.96155])

        #
        Rt_controller2world = tf.transformations.concatenate_matrices(t_controller2world, R_controller2world)

        t_end2world = tf.transformations.translation_matrix(Rt_controller2world.dot(t_end2controller)[:3, :].reshape(3, ))
        R_end2world = R_controller2world.dot(R_0)
        Rt_end2world = tf.transformations.concatenate_matrices(t_end2world, R_end2world)

        Rt_board2world = tf.transformations.concatenate_matrices(t_board2world, R_board2world)
        Rt_world2board = np.linalg.inv(Rt_board2world)

        Rt_cam2board = tf.transformations.concatenate_matrices(t_cam2board, R_cam2board)
        Rt_board2cam = np.linalg.inv(Rt_cam2board)

        Rt_end2cam = tf.transformations.concatenate_matrices(Rt_board2cam, Rt_world2board, Rt_end2world)

        # # >>>>
        # ps.header.stamp = rospy.Time.now()
        # # ps.header.frame_id = '/world_vive'
        # ps.header.frame_id = '/camera_rgb_optical_frame'
        # # ps.child_frame_id = '/test'

        # rt = tf.transformations.concatenate_matrices(Rt_board2cam, Rt_world2board, Rt_end2world)
        # # rt = tf.transformations.concatenate_matrices(Rt_board2world, np.linalg.inv(Rt_end2world))
        # # rt = tf.transformations.concatenate_matrices(Rt_world2board)
        # print(rt)
        # pos = rt[:3, 3].copy()
        # # print(pos)
        # rt[:3, 3] = 0
        # # print(rt)
        # quat = tf.transformations.quaternion_from_matrix(rt)

        # # ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos.dot([-0.236220294307, -1.2843890696, -3.01909357602])
        # # ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos.dot([[0], [0], [0], [1]])
        # ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos
        # # ps.transform.translation.x, ps.transform.translation.y, ps.transform.translation.z = tf.transformations.translation_from_matrix(pos)
        # ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = quat
        # # ps.transform.rotation.x, ps.transform.rotation.y, ps.transform.rotation.z, ps.transform.rotation.w = tf.transformations.quaternion_from_matrix(quat)
        # # TrackerFrame.transmitter.sendTransformMessage(ps)
        # # TrackerFrame.transmitter.sendTransform(pos, quat, ps.header.stamp, ps.child_frame_id, ps.header.frame_id)
        # TrackerFrame.poser.publish(ps)

        return Rt_end2cam

    def solvePointsSVD(self, A, B):
        A = np.array(A, dtype=np.float128).reshape(3, -1)
        B = np.array(B, dtype=np.float128).reshape(3, -1)

        # n = A.shape[1]

        # compute weighted centers of both sets
        meanA = np.mean(A, axis=1)[..., np.newaxis]
        meanB = np.mean(B, axis=1)[..., np.newaxis]

        # normalize (center) both sets
        A_normalized = A - meanA
        B_normalized = B - meanB

        # Compute S = A * W * B.T
        # S = A_normalized * B_normalized.T
        S = np.dot(A_normalized, B_normalized.T)

        # Do SVD to decompose S
        U, e, Vt = np.linalg.svd(S.astype(np.float64))

        # Now, it should be possible to compute R = V * DE * U.T
        R_optim = np.dot(Vt.T, U.T)  #  this one is B = R'*A
        # R_optim = U * Vt  # this should be A = R * B - but I am not sure if that is actually correct

        # Because B = R * A + t, we can rearange to get t = B - R * A and therefore:
        # optimal translation vector should be computed as t = center_B - R * center_A
        t_optim = meanB - np.dot(R_optim, meanA)

        return R_optim, t_optim

    def solvePointsCovar(self, A, B):
        A = np.matrix(A, dtype=np.float128).reshape(3, -1)
        B = np.matrix(B, dtype=np.float128).reshape(3, -1)

        # compute weighted centers of both sets
        meanA = np.mean(A, axis=1)
        meanB = np.mean(B, axis=1)

        # normalize (center) both sets
        A_normalized = A - meanA
        B_normalized = B - meanB

        # Compute M = A * W * B.T
        M = A_normalized * B_normalized.T

        # Compute R
        R_optim = M / sqrtm(M.T * M)

        # Because B = R * A + t, we can rearange to get t = B - R * A and therefore:
        # optimal translation vector should be computed as t = center_B - R * center_A
        t_optim = meanB - R_optim * meanA

        return R_optim, t_optim

    def best_fit_transform(self, A, B):
        '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
        A: Nxm numpy array of corresponding points
        B: Nxm numpy array of corresponding points
        Returns:
        T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        R: mxm rotation matrix
        t: mx1 translation vector
        '''
        # A = np.matrix(A, dtype=np.float128).reshape(-1, 3)
        # B = np.matrix(B, dtype=np.float128).reshape(-1, 3)
        A = A.T
        B = B.T

        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R, centroid_A.T)

        return R, t

    def solvePosition(self):
        """ Main function that computes the transformation between the tracked frame
        and the world_frame.
        """
        def func(beta, x):
            """ implicit definition of the sphere """
            return (x[0] - beta[0])**2 + (x[1] - beta[1])**2 + (x[2] - beta[2])**2 - beta[3]**2

        def calc_R(xc, yc, zc):
            """ calculate the distance of each 3D points from the center (xc, yc, zc) """
            return np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2)

        """
        A - points in the world coordinate system
        B - points in the tracked coordinate system
        w - weights for B points based on the goodnes of fit
        tip_ts - translation from tracked frame to the estimated tool endpoint
        ptidx - indices of the collected points
        """
        A, B, w, tip_ts, ptidx = [], [], [], [], []
        collectedSets = 0
        # enumerates through the chessboard points
        for i, (c, r) in enumerate(np.ndindex(CHECKERBOARD_SIZE[1], CHECKERBOARD_SIZE[0])):
            gpoints = self.posePoints[r, c]  # check the collected points for this corner on the board
            if not gpoints:  # continue if no points were collected for this corner
                continue
            collectedSets += 1
            grotations = self.poseOrientations[r, c]
            ptidx.append(i)

            # convert ROS Point to numpy array
            points = np.array([(pt.x, pt.y, pt.z) for pt in gpoints], dtype=np.float64)
            # separate x, y, z coordinates
            x, y, z = points[:, 0], points[:, 1], points[:, 2]

            # find centers of collected points
            x_m = np.mean(x)
            y_m = np.mean(y)
            z_m = np.mean(z)

            # initial guess for parameters (mean of the collected points)
            R_m = calc_R(x_m, y_m, z_m).mean()
            beta0 = [x_m, y_m, z_m, R_m]

            # for implicit function :
            #       data.x contains both coordinates of the points (data.x = [x, y, z])
            #       data.y is the dimensionality of the response
            data = np.row_stack([x, y, z])
            lsc_data = sp.odr.Data(data, y=1)
            lsc_model = sp.odr.Model(func, implicit=True)
            lsc_odr = sp.odr.ODR(lsc_data, lsc_model, beta0)
            lsc_out = lsc_odr.run()

            xc, yc, zc, R = lsc_out.beta
            Ri = calc_R(xc, yc, zc)
            residu = sum((Ri - R)**2)

            # publish the estimated center point to ROS
            marker = Marker()
            marker.action = Marker.ADD
            marker.header.frame_id = TRACKED_FRAME
            marker.header.stamp = rospy.Time.now()
            marker.id = self.markerIDs[r, c] + 10000
            marker.text = "alfa"
            marker.type = Marker.CUBE
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = xc, yc, zc
            marker.scale.x = marker.scale.y = marker.scale.z = 0.015
            marker.color.r, marker.color.g, marker.color.b = self.markerColors[r, c]
            marker.color.a = 0.7
            self.markerPublisher.publish(marker)

            # Append the right coordinates for A, B and weight
            # A -> Points in the board coordinate system
            # B -> Points in the tracked coordinate system, i.e. the calculated centers of spheres
            A.append(self.camera.boardPoseCalculator.objp[i])
            B.append([xc, yc, zc])
            w.append(1 - residu)

            print('Point {} with grid coordinates: [{}, {}]; # of samples = {} and residual = {}'.format(
                i, r, c, len(points), colorValue(residu, 1e-3, 1e-4)))

            # pBs are "points B" translated so that the individual tracked points are the center of the coordinate system
            pBs = np.pad(np.matrix([xc, yc, zc]) - points, ((0, 0), (0, 1)), 'constant')
            # Now we need to rotate the pBs so that the orientations of the tracked points are aligned with the axis of the coordinate system
            tpB = np.array([(np.matrix(tf.transformations.quaternion_matrix([rt.x, rt.y, rt.z, rt.w])).T * pB.reshape(4, 1))[:3] for (rt, pB) in zip(grotations, pBs)], dtype=np.float64).squeeze()
            # Finally, find the average translation and append it to the list (ideally, all the values in the list will be the same)
            static_translation = np.median(tpB, axis=0)
            tip_ts.append(static_translation)
            print("Tracker to endpoint translation = ", static_translation)

        A = np.array(A, dtype=np.float64).T  # board points
        B = np.array(B, dtype=np.float64).T  # tracked frame points

        # compute the plane - the estimated centers should lie on a plane
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(B.T)
        plane, inliers = pcd.segment_plane(distance_threshold=0.001, ransac_n=max(3, int(np.ceil(np.sqrt(B.shape[1])))), num_iterations=10000)
        a, b, c, d = plane
        # B = B + np.r_[a, b, c][:, np.newaxis] * TRACKER_PLANE_OFFSET  # offset in Z axis
        goodA, goodB, goodIdx = A[:, inliers], B[:, inliers], np.r_[ptidx][inliers]  # filter out the outliers

        rospy.loginfo("Found {} inliers out of {} points total after plane fitting. Pose optimization will only use points {}.".format(len(inliers), B.shape[1], np.array2string(goodIdx)))

        # Solve the transformation B = R * A + t
        self.R_optim, self.t_optim = self.best_fit_transform(goodA, goodB)

        # convert to homogenious coords
        Ah = np.pad(goodA, ((0, 1), (0, 0)), "constant", constant_values=1)
        Bh = np.pad(goodB, ((0, 1), (0, 0)), "constant", constant_values=1)
        self.t_optim = self.t_optim[:, np.newaxis]
        self.R_optim = np.pad(self.R_optim, ((0, 1), (0, 1)), 'constant')
        self.R_optim[-1, -1] = 1

        # compute tracked frame -> board transformation, i.e. B -> A
        self.board2TrackerTransform = tf3.affines.compose(self.t_optim.ravel(), self.R_optim[:3, :3], np.ones(3))
        self.tracker2boardTransform = np.linalg.inv(self.board2TrackerTransform)

        rospy.loginfo('Calibration complete.')
        residualMagnitude = np.linalg.norm(self.tracker2boardTransform.dot(Bh) - Ah, axis=0)
        rospy.loginfo('\nResidual from transformation optimization:\nmin = {}\naverage = {}\nmax = {}'.format(
            colorValue(residualMagnitude.min(), 1e-3, 1e-4), colorValue(residualMagnitude.mean(), 1e-3, 1e-4), colorValue(residualMagnitude.max(), 1e-3, 1e-4)))
        residualInverseMagnitude = np.linalg.norm(np.linalg.inv(self.tracker2boardTransform).dot(Ah) - Bh, axis=0)
        rospy.loginfo('\nResidual from INVERSE transformation:\nmin = {}\naverage = {}\nmax = {}'.format(
            colorValue(residualInverseMagnitude.min(), 1e-3, 1e-4), colorValue(residualInverseMagnitude.mean(), 1e-3, 1e-4), colorValue(residualInverseMagnitude.max(), 1e-3, 1e-4)))

        # self.qR_optim = tf.transformations.quaternion_from_matrix(self.R_optim)
        # self.qR_optim = self.qR_optim / np.linalg.norm(self.qR_optim)  # normalize to get unit quaternion
        # self.qR_optim = tf3.quaternions.mat2quat(tf3.quaternions.quat2mat(self.qR_optim))

        # Publish the static transform from the tracked point to the tip
        self.trackedToTargetTranslation = np.median(tip_ts, axis=0)
        rospy.loginfo('Static translation from the tracked point to the center = {}'.format(self.trackedToTargetTranslation))
        self.sendStaticTransformation([0, 0, 0, 1], self.trackedToTargetTranslation)

        # Prepare TF message
        t, R, Z, S = tf3.affines.decompose(self.board2TrackerTransform)
        quat = tf.transformations.quaternion_from_matrix(self.board2TrackerTransform)
        rospy.loginfo('\nThe transformation from board to tracked frame:\n\tt = {}\n\tR = {}'.format(np.array2string(t), np.array2string(np.rad2deg(tf3.euler.quat2euler(quat)))))

        self.transformStampedMessage = TransformStamped()
        self.transformStampedMessage.header.frame_id = TRACKED_FRAME
        self.transformStampedMessage.child_frame_id = WORLD_FRAME
        self.transformStampedMessage.transform.translation.x = t[0]
        self.transformStampedMessage.transform.translation.y = t[1]
        self.transformStampedMessage.transform.translation.z = t[2]
        self.transformStampedMessage.transform.rotation.x = quat[0]
        self.transformStampedMessage.transform.rotation.y = quat[1]
        self.transformStampedMessage.transform.rotation.z = quat[2]
        self.transformStampedMessage.transform.rotation.w = quat[3]

        #####
        np.save('translate_ctrl_to_endpoint', self.trackedToTargetTranslation)
        #####

        self.isCalibrated = True
        self.phase = self.PHASE_DONE
        self.buttonCollect.Enable()
        self.soundWin.play()
        wx.CallLater(self.UPDATE_INTERVAL, self.sendTransformation)

    def sendStaticTransformation(self, quat, t):
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_transformStamped = TransformStamped()

        static_transformStamped.header.stamp = rospy.Time.now()
        static_transformStamped.header.frame_id = self.trackedTopic
        static_transformStamped.child_frame_id = 'end_point'

        static_transformStamped.transform.translation.x = t[0]
        static_transformStamped.transform.translation.y = t[1]
        static_transformStamped.transform.translation.z = t[2]

        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]

        broadcaster.sendTransform(static_transformStamped)

    def sendTransformation(self):
        self.transformStampedMessage.header.stamp = rospy.Time.now()
        self.tfBroadcaster.sendTransformMessage(self.transformStampedMessage)
        # self.tfBroadcaster.sendTransform(t, q, rospy.Time.now(), TRACKED_FRAME, WORLD_FRAME)
        # self.tfBroadcaster.sendTransform(self.t_optim, self.qR_optim, rospy.Time.now(), WORLD_FRAME, TRACKED_FRAME)
        # self.tfBroadcaster.sendTransform(self.t_optim, self.qR_optim, rospy.Time.now(), TRACKED_FRAME, WORLD_FRAME)
        if self.isCalibrated:
            wx.CallLater(self.UPDATE_INTERVAL, self.sendTransformation)

    def trackedTopicCallback(self, msg):
        if self.phase & self.PHASE_WAITING_POSE_TOPIC:
            self.phase &= ~self.PHASE_WAITING_POSE_TOPIC
            wx.CallAfter(self.checkTrackedTopic.SetValue, True)
            wx.CallAfter(self.checkTrackedTopic.SetForegroundColour, wx.GREEN)
            self.topicFreshnessCounter = rospy.Time.now()
            if self.phase & self.PHASE_WAITING_IMAGE_TOPIC:
                wx.CallAfter(self.gauge.SetValue, 0)
                wx.CallAfter(self.infoText.ChangeValue, 'Messages from the tracked topic received, waiting for images...')
        else:
            if (rospy.Time.now() - self.topicFreshnessCounter).secs > 3:
                if (rospy.Time.now() - self.topicFreshnessCounter).secs > 10:
                    wx.CallAfter(self.checkTrackedTopic.SetForegroundColour, wx.RED)
                else:
                    wx.CallAfter(self.checkTrackedTopic.SetForegroundColour, wx.YELLOW)
            else:
                wx.CallAfter(self.checkTrackedTopic.SetForegroundColour, wx.GREEN)
            self.topicFreshnessCounter = rospy.Time.now()

        if self.phase & self.PHASE_POINT_GATHERING:
            self.points.append(msg.pose.position)
            self.orientations.append(msg.pose.orientation)
            self.topicFrameID = msg.header.frame_id
            n = len(self.points)
            if n < self.GAUGE_RANGE:
                wx.CallAfter(self.gauge.SetValue, n)
        elif self.phase & self.PHASE_DONE:
            self.trackedPoint = msg.pose
            # trackedPoint = np.array([self.trackedPoint.position.x, self.trackedPoint.position.y, self.trackedPoint.position.z], dtype=np.float64).reshape(3, 1)
            # rmat = np.array(tf.transformations.quaternion_matrix([self.trackedPoint.orientation.x, self.trackedPoint.orientation.y, self.trackedPoint.orientation.z, self.trackedPoint.orientation.w]), dtype=np.float64)
            # centerPoint = (trackedPoint + rmat[:3, :3] * self.trackedToTargetTranslation.reshape(3, 1)).astype(np.float64).reshape(1, 3)
            # rmat = rmat * np.array(tf.transformations.euler_matrix(0, -np.pi / 2, 0))
            # ps = PoseStamped()
            # ps.header.stamp = msg.header.stamp
            # ps.header.frame_id = msg.header.frame_id
            # ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = centerPoint.T
            # quat = tf.transformations.quaternion_from_matrix(rmat)
            # ps.pose.orientation = Quaternion(*quat)
            # self.posePublisher.publish(ps)

    def updateImage(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg).copy()
        # img = self.bridge.compressed_imgmsg_to_cv2(msg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        if self.phase & self.PHASE_WAITING_IMAGE_TOPIC:
            self.phase = self.phase & ~self.PHASE_WAITING_IMAGE_TOPIC | self.PHASE_IMAGE_CALIBRATION
            self.camera.boardPoseCalculator.resetBoardOrientation()
            wx.CallAfter(self.infoText.ChangeValue, 'Received images from camera, please wait while the system calibrates the camera position...')

        if self.phase & self.PHASE_IMAGE_CALIBRATION:
            # >>> Phase 1: Estimate the camera position in relation to the checkerboard
            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None, cv2.CALIB_CB_FAST_CHECK)

            if ret:
                # Find the rotation and translation vectors.
                rvec = self.rvecsBuffer[-1] if len(self.rvecsBuffer) > 0 else None
                tvec = self.tvecsBuffer[-1] if len(self.tvecsBuffer) > 0 else None

                ret, rvec, tvec, corners2 = self.camera.boardPoseCalculator.estimateBoardPose(gray, rvec, tvec)

                self.rvecsBuffer.append(rvec)
                self.tvecsBuffer.append(tvec)
                wx.CallAfter(self.gauge.SetValue, int(self.gauge.GetRange() * (float(len(self.rvecsBuffer)) / CALIB_MAX_TRANSFORM_VECTORS)))

                if len(self.rvecsBuffer) == CALIB_MAX_TRANSFORM_VECTORS:
                    self.rvec = np.median(self.rvecsBuffer, axis=0)
                    self.tvec = np.median(self.tvecsBuffer, axis=0)
                    self.cam_Rmat, _ = cv2.Rodrigues(self.rvec)
                    self.board2camTransform = tf3.affines.compose(self.tvec.ravel(), self.cam_Rmat, np.ones(3))
                    self.corners = corners2
                    self.phase = self.phase & ~self.PHASE_IMAGE_CALIBRATION | self.PHASE_TRACKING
                    self.markerPoints, jac = cv2.projectPoints(self.camera.boardPoseCalculator.objp, self.rvec, self.tvec, self.camera.camMatrix, None)
                    self.markerPoints = np.squeeze(self.markerPoints.astype(np.int16))
                    self.imgpts, jac = cv2.projectPoints(self.area, self.rvec, self.tvec, self.camera.camMatrix, None)
                    self.buttonLoad.Enable()
                    self.soundHorn.play()
            else:
                corners2 = corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners2, ret)
        elif self.phase & self.PHASE_TRACKING:
            img = self.__drawArea(img, self.imgpts)
            markerPt, jac = cv2.projectPoints(self.camera.boardPoseCalculator.objp[self.currentPoint].reshape(1, 3), self.rvec, self.tvec, self.camera.camMatrix, None)
            markColor = (0, 255, 140, 250) if self.phase & self.PHASE_POINT_GATHERING else (160, 160, 230, 200)
            img = cv2.drawMarker(img, tuple(markerPt.ravel().astype(np.int16)), markColor, cv2.MARKER_DIAMOND, 64, self.camera.lineThickness * 3)
            collectedSets = 0
            for qi in np.ndindex(self.posePoints.shape):
                if not self.posePoints[qi]:
                    continue
                collectedSets += 1
                cfq = len(self.posePoints[qi]) / float(self.GAUGE_RANGE)
                if cfq > 1:
                    cfq = 1
                markColor = (255 * (1 - cfq), 255 * cfq, 0, 128)
                idx = qi[0] + self.posePoints.shape[0] * qi[1]
                img = cv2.drawMarker(img, tuple(self.markerPoints[idx]), markColor, cv2.MARKER_SQUARE, 10, self.camera.lineThickness)
            if collectedSets > 2:
                self.buttonCalib.Enable()
            else:
                self.buttonCalib.Disable()
        elif self.phase & self.PHASE_DONE:
            # >>> Phase 5: Success!
            img = self.__drawArea(img, self.imgpts)
            #  print(self.trackedPoint.x)
            if self.trackedPoint:
                # parent_frame, target_frame = self.camera.cinfo.frame_id, self.trackedTopic
                # parent_frame, target_frame = '/world', '/world_vive'
                # time = self.tfListener.getLatestCommonTime(parent_frame, target_frame)
                # try:
                #     self.tfListener.waitForTransform(parent_frame, target_frame, time, 2000)
                #     transform = self.tfListener.lookupTransform(parent_frame, target_frame, time)
                #     print('tf success', transform)
                # except:
                #     print('tf timeout')
                trackedPoint = np.r_[self.trackedPoint.position.x, self.trackedPoint.position.y, self.trackedPoint.position.z, 1].astype(np.float64).reshape(-1, 1)
                # tpoint = np.dot(self.R_optim[:3, :3].T, (trackedPoint - self.t_optim).astype(np.float64))
                # rmat = np.matrix(tf.transformations.quaternion_matrix([self.trackedPoint.orientation.x, self.trackedPoint.orientation.y, self.trackedPoint.orientation.z, self.trackedPoint.orientation.w]), dtype=np.float64)[:3, :3]
                # cpoint = np.dot(self.R_optim[:3, :3].T, ((trackedPoint + np.dot(rmat, self.trackedToTargetTranslation.reshape(3, 1))) - self.t_optim).astype(np.float64))

                # point = self.Rt_end2cam([self.trackedPoint.orientation.x, self.trackedPoint.orientation.y, self.trackedPoint.orientation.z, self.trackedPoint.orientation.w], [self.trackedPoint.position.x, self.trackedPoint.position.y, self.trackedPoint.position.z]).dot([[0], [0], [0], [1]])
                # print(point)
                # u, v, w = self.camera.camMatrix.dot(point)
                # u, v, w = self.camera.camMatrix.dot(tpoint)
                u, v, w = self.camera.camMatrix.dot(self.board2camTransform.dot(self.tracker2boardTransform.dot(trackedPoint))[:3, :])
                mkpt = np.array([u / w, v / w])
                print(mkpt)
                # cpoint = point[:3, :].reshape(1, 3)

                # markerPts, jac = cv2.projectPoints(np.row_stack((tpoint, cpoint)), self.rvec, self.tvec, self.camera.camMatrix, None)
                # img = cv2.drawMarker(img, tuple(markerPts[0].ravel().astype(np.int16)), (64, 255, 255, 255), cv2.MARKER_TRIANGLE_DOWN, 10, self.camera.lineThickness)
                # img = cv2.drawMarker(img, tuple(markerPts[1].ravel().astype(np.int16)), (64, 255, 255, 255), cv2.MARKER_TILTED_CROSS, 10, self.camera.lineThickness)
                img = cv2.drawMarker(img, tuple(mkpt.ravel().astype(np.int16)), (64, 255, 128, 255), cv2.MARKER_TILTED_CROSS, 10, self.camera.lineThickness)

#        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        wx.CallAfter(self.imageView.update, img)

    def onKeyDownOverButton(self, event):
        key = event.GetKeyCode()
        if key not in [wx.WXK_SPACE, wx.WXK_RETURN]:
            event.Skip()
            return
        obj = event.GetEventObject()
        if type(obj) is wx.TextCtrl and not obj.GetWindowStyleFlag() & wx.TE_READONLY:
            if key == wx.WXK_ESCAPE:
                self.buttonLoad.SetFocus()
            else:
                event.Skip()

    def onKeyDown(self, key):
        # Pynput is great but captures events when window is not focused. The following condition prevents that.
        if not self.IsActive():
            return
        if not self.phase & self.PHASE_TRACKING:
            return
        if not self.phase & self.PHASE_POINT_GATHERING:
            if key == pynput.keyboard.Key.left:
                self.currentPoint -= 1
            elif key == pynput.keyboard.Key.right:
                self.currentPoint += 1
            elif key == pynput.keyboard.Key.up:
                self.currentPoint -= CHECKERBOARD_SIZE[0]
            elif key == pynput.keyboard.Key.down:
                self.currentPoint += CHECKERBOARD_SIZE[0]
            elif key == pynput.keyboard.Key.backspace:
                x, y, z = self.camera.boardPoseCalculator.gridpoints[self.currentPoint]
                r, c = int(x), int(y)
                self.posePoints[r, c] = None
                self.poseOrientations[r, c] = None
            elif key == pynput.keyboard.Key.space:
                wx.CallAfter(self.startTrackingPoints)
        else:
            if key == pynput.keyboard.Key.space:
                self.phase &= ~self.PHASE_POINT_GATHERING
                x, y, z = self.camera.boardPoseCalculator.gridpoints[self.currentPoint]
                r, c = int(x), int(y)

                self.posePoints[r, c] = self.points
                self.poseOrientations[r, c] = self.orientations

                marker = Marker()
                marker.action = Marker.ADD
                marker.header.frame_id = TRACKED_FRAME
                marker.header.stamp = rospy.Time.now()
                marker.id = self.markerIDs[r, c]
                marker.type = Marker.SPHERE_LIST
                marker.points = self.points
                marker.scale.x = marker.scale.y = marker.scale.z = 0.001
                marker.color.r, marker.color.g, marker.color.b = self.markerColors[r, c]
                marker.color.a = 0.9
                self.markerPublisher.publish(marker)

                self.currentPoint += 1
                self.soundBeep.play()

        if self.currentPoint < 0:
            self.currentPoint = 0
        elif self.currentPoint >= len(self.camera.boardPoseCalculator.objp):
            self.currentPoint = len(self.camera.boardPoseCalculator.objp) - 1

    def onImageFeedButton(self, event):
        print("Not implemented, yet!")

    def onLoadButton(self, event):
        with wx.FileDialog(self, "Select calibration file", defaultDir=os.path.join(self.packagePath, 'data'), wildcard="Numpy 'NPZ' files (*.npz)|*.npz", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return     # the user changed their mind
            # Proceed loading the file chosen by the user
            pathname = fileDialog.GetPath()
            self.phase = self.PHASE_TRACKING
            data = np.load(pathname, allow_pickle=True)
            self.posePoints = data['points']
            self.poseOrientations = data['orientations']

    def onCollectButton(self, event):
        if self.phase & self.PHASE_DONE:
            self.buttonCollect.Disable()
            self.isCalibrated = False
            self.phase = self.phase & ~self.PHASE_DONE | self.PHASE_TRACKING

    def onSaveButton(self, event):
        """On pressing the "Save points" button, saves the collected points into Numpy's NPZ files.
        The file name is automatically generated as 'points' + <current_datetime> + '.npz'
        and the file is saved in the current packages "data" folder (make sure it exists).
        Arguments:
            event {wx.Event}
        """

        dt = datetime.datetime.now()
        fname = os.path.join(self.packagePath, 'data', dt.strftime('points_%Y%m%d_%H%M%S') + '.npz')
        np.savez(fname, points=self.posePoints, orientations=self.poseOrientations)
        print('Points saved! Filename: {}'.format(fname))

    def onCalibButton(self, event):
        self.solvePosition()

    def onSampleColorReceived(self, _, ogrid):
        if not self.phase & self.PHASE_OBJECT_MARKING:
            return
        sample = self.camera.getPixels(ogrid).reshape((1, -1, 3))

        self.contourAreaSizeThreshold = np.max([self.CONTOUR_AREA_SIZE_THRESHOLD, sample.shape[0]])
        sampleLAB = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB)
        sampleHSV = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)

        self.colorTrackingParams = ColorTrackingParams()

        labMin = np.min(sampleLAB, axis=1) * (1 - self.LAB_VALUE_BLUR)
        labMin[0, 0] = 26
        labMax = np.max(sampleLAB, axis=1) * (1 + self.LAB_VALUE_BLUR)
        labMax[0, 0] = 240
        self.colorTrackingParams.addParams(labMin=labMin, labMax=labMax)

        hsvMin = np.clip(np.min(sampleHSV, axis=1) * (1 - self.HSV_VALUE_BLUR), np.iinfo(sampleHSV.dtype).min, np.iinfo(sampleHSV.dtype).max)
        hsvMax = np.clip(np.max(sampleHSV, axis=1) * (1 + self.HSV_VALUE_BLUR), np.iinfo(sampleHSV.dtype).min, np.iinfo(sampleHSV.dtype).max)
        self.colorTrackingParams.addParams(hsvMin=hsvMin, hsvMax=hsvMax)
        print(list(self.colorTrackingParams.__dict__.iteritems()))

    def __drawArea(self, img, imgpts):
        origin = tuple(imgpts[0].ravel())
        img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (255, 0, 0, 128), self.camera.lineThickness)
        img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0, 128), self.camera.lineThickness)
        img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (255, 0, 0, 128), self.camera.lineThickness)
        img = cv2.line(img, tuple(imgpts[3].ravel()), origin, (0, 255, 0, 128), self.camera.lineThickness)
        img = cv2.arrowedLine(img, origin, tuple(imgpts[4].ravel()), (0, 0, 255, 128), self.camera.lineThickness)
        return img

    def startTrackingPoints(self):
        self.points = []
        self.orientations = []
        self.phase |= self.PHASE_POINT_GATHERING
        self.soundCoin.play()

    def onClose(self, event):
        print('>>> CLOSING TRACKER FRAME <<<')
        self.keyDaemon.stop()
        self.imageSubscriber.unregister()
        self.trackingSubscriber.unregister()
        self.DestroyLater()


class Camera(object):
    PHASE_CAM_WAITING_CAM_INFO = 1
    PHASE_CAM_WAITING_IMAGE = 2
    PHASE_CAM_EXT_CALIB = 4
    PHASE_CAM_POSE = 8
    PHASE_CAM_COMPLETE = 16

    UPDATE_CHECK_STATUS = 1
    UPDATE_CHECK_PROGRESS = 2

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # objp = np.zeros((CHECKERBOARD_SIZE[1] * CHECKERBOARD_SIZE[0], 3), np.float64)
    # objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0] * CHECKERBOARD_CELL_SIZE:CHECKERBOARD_CELL_SIZE, 0:CHECKERBOARD_SIZE[1] * CHECKERBOARD_CELL_SIZE:CHECKERBOARD_CELL_SIZE].T.reshape(-1, 2)
    axis = np.float64([[0, 0, 0], [3 * CHECKERBOARD_CELL_SIZE, 0, 0], [0, 3 * CHECKERBOARD_CELL_SIZE, 0], [0, 0, 3 * CHECKERBOARD_CELL_SIZE]]).reshape(-1, 3)
    wx.TextCtrl

    def __init__(self, updateCam_cb, autoCalibrate=False):
        self.ID = uuid.uuid4()
        self.updateCam_cb = updateCam_cb
        self.frame_name = 'Frame not assigned yet, ID:{}'.format(self.ID)
        self.bridge = CvBridge()
        self.autoCalibrate = autoCalibrate

        self.phase = self.PHASE_CAM_WAITING_CAM_INFO | self.PHASE_CAM_WAITING_IMAGE
        self.rvecsBuffer = []
        self.tvecsBuffer = []
        self.isPoseCalibrated = False
        self.rate = rospy.Rate(1)
        self.tfBroadcaster = tf.TransformBroadcaster()
        self.image = None
        self.boardPoseCalculator = None

    def calibrate(self):
        self.rvecsBuffer = []
        self.tvecsBuffer = []
        self.isPoseCalibrated = False
        wx.CallAfter(self.updateCam_cb, self, self.UPDATE_CHECK_PROGRESS | self.UPDATE_CHECK_STATUS)
        if not (self.phase & (self.PHASE_CAM_WAITING_CAM_INFO | self.PHASE_CAM_WAITING_IMAGE)):
            self.boardPoseCalculator.resetBoardOrientation()
            self.phase = self.PHASE_CAM_EXT_CALIB

    def setCinfoCB(self, cinfoTopic):
        try:
            self.cinfoSubscriber.unregister()
        except:  # noqa
            pass  # subscriber hasnt been created or is None, but no one cares
        self.cinfoTopic = cinfoTopic
        rospy.loginfo('Subscribing to camera info on topic {}.'.format(cinfoTopic))
        self.cinfoSubscriber = rospy.Subscriber(cinfoTopic, rostopic.get_topic_class(cinfoTopic)[0], self.__cameraInfo_cb)
        self.phase |= self.PHASE_CAM_WAITING_CAM_INFO
        wx.GetApp().frame.FindWindowById(self.buttonCinfoTopic).SetToolTip(wx.ToolTip(self.cinfoTopic))
        wx.CallAfter(self.updateCam_cb, self, flag=self.UPDATE_CHECK_STATUS)

    def setImageCB(self, imageTopic):
        # Subscribe to the image topic
        try:
            self.imageSubscriber.unregister()
        except:  # noqa
            pass  # subscriber hasnt been created or is None, but no one cares
        self.imageTopic = imageTopic
        rospy.loginfo('Subscribing to images on topic {}.'.format(imageTopic))
        self.imageSubscriber = rospy.Subscriber(imageTopic, rostopic.get_topic_class(imageTopic)[0], callback=self.image_cb, queue_size=1)
        self.phase |= self.PHASE_CAM_WAITING_IMAGE
        wx.GetApp().frame.FindWindowById(self.buttonImageTopic).SetToolTip(wx.ToolTip(self.imageTopic))
        wx.CallAfter(self.updateCam_cb, self, flag=self.UPDATE_CHECK_STATUS)

    def __drawAxis(self, img, imgpts):
        origin = tuple(imgpts[0].ravel().astype(np.int16))
        img = cv2.arrowedLine(img, origin, tuple(imgpts[1].ravel().astype(np.int16)), (0, 0, 255), self.lineThickness)
        img = cv2.arrowedLine(img, origin, tuple(imgpts[2].ravel().astype(np.int16)), (0, 255, 0), self.lineThickness)
        img = cv2.arrowedLine(img, origin, tuple(imgpts[3].ravel().astype(np.int16)), (255, 0, 0), self.lineThickness)
        return img

    def __cameraInfo_cb(self, msg):
        self.cinfo = msg
        self.frame_name = self.cinfo.header.frame_id
        if self.isPoseCalibrated:
            if self.rate.remaining() <= rospy.Duration():
                rot_mat = np.pad(cv2.Rodrigues(self.rvecs)[0], ((0, 1), (0, 1)), 'constant')
                rot_mat[3, 3] = 1

                # compute inverse transformation, so that we have T from camera to world
                rotation = tf.transformations.quaternion_from_matrix(rot_mat.T)
                translation = np.row_stack((self.tvecs, [0]))
                translation = rot_mat.T.dot(translation * -1)
                translation = translation[:3]

                self.tfBroadcaster.sendTransform(translation, rotation, rospy.Time.now(), msg.header.frame_id, WORLD_FRAME)
                self.rate.sleep()
                return
        if self.phase & self.PHASE_CAM_WAITING_CAM_INFO:
            self.phase &= ~self.PHASE_CAM_WAITING_CAM_INFO
            self.camMatrix = np.float64(self.cinfo.K).reshape((3, 3))
            self.distMatrix = np.float64(self.cinfo.D).reshape((1, 5))
            self.lineThickness = np.ceil(np.max([self.cinfo.width, self.cinfo.height]) / 1000.).astype(np.int16) + 1
            self.boardPoseCalculator = PoseEstimator(self.camMatrix, self.distMatrix)
            wx.CallAfter(self.updateCam_cb, self, flag=self.UPDATE_CHECK_STATUS)
        if self.autoCalibrate and not self.phase & self.PHASE_CAM_WAITING_IMAGE:
            self.calibrate()

    def getPixels(self, ogrid):
        return self.image[ogrid]

    def image_cb(self, msg):
        if self.phase & self.PHASE_CAM_WAITING_IMAGE:
            self.phase &= ~self.PHASE_CAM_WAITING_IMAGE
            wx.CallAfter(self.updateCam_cb, self, flag=self.UPDATE_CHECK_STATUS)
            if self.autoCalibrate and not self.phase & self.PHASE_CAM_WAITING_CAM_INFO:
                self.calibrate()

        img = self.bridge.imgmsg_to_cv2(msg).copy()  # copy has to be here otherwise this bridge uses the same image in all callbacks
        # img = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.image = img.copy()
        if self.phase == self.PHASE_CAM_EXT_CALIB:
            if (len(img.shape) == 3) and (img.shape[2] == 3):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None, cv2.CALIB_CB_FAST_CHECK)

            if ret:
                # Find the rotation and translation vectors.
                rvecs = self.rvecsBuffer[-1] if len(self.rvecsBuffer) > 0 else None
                tvecs = self.tvecsBuffer[-1] if len(self.tvecsBuffer) > 0 else None

                ret, rvecs, tvecs, corners2 = self.boardPoseCalculator.estimateBoardPose(gray, rvecs, tvecs)
                self.rvecsBuffer.append(rvecs)
                self.tvecsBuffer.append(tvecs)
                wx.CallAfter(self.updateCam_cb, self, self.UPDATE_CHECK_PROGRESS)
                if len(self.rvecsBuffer) == CALIB_MAX_TRANSFORM_VECTORS:
                    self.rvecs = np.median(self.rvecsBuffer, axis=0)
                    self.tvecs = np.median(self.tvecsBuffer, axis=0)
                    self.R_mat = cv2.Rodrigues(self.rvecs)
                    self.phase = self.PHASE_CAM_POSE
                    self.isPoseCalibrated = True
                    wx.CallAfter(self.updateCam_cb, self, self.UPDATE_CHECK_STATUS)
            else:
                corners2 = corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners2, ret)
        elif self.phase == self.PHASE_CAM_POSE:
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(self.axis, self.rvecs, self.tvecs, self.camMatrix, None)
            img = self.__drawAxis(img, imgpts)
        elif self.phase == self.PHASE_CAM_COMPLETE:
            pass

        wx.CallAfter(self.updateCam_cb, self, img=img)

    def destroy(self):
        self.imageSubscriber.unregister()
        self.cinfoSubscriber.unregister()


class CameraPoseCalibration(wx.App):
    camControlNames = ['textStatus', 'buttonImageTopic', 'checkImageTopic', 'buttonCinfoTopic', 'checkCinfoTopic', 'buttonCalib', 'gaugeCalib', 'checkCalib', 'buttonInfo', 'buttonView']

    def OnInit(self):
        # ROS initialization
        rospy.init_node("pose_estimator")
        rospack = rospkg.RosPack()
        self.packagePath = rospack.get_path('cam_calib')

        self.cameras = []
        self.camDict = {}
        self.imageViewCamID = None
        self.whiteRect = None

        # >>> Setup GUI <<<
        self.frame = wx.Frame(None, title="Calibrator", size=(32, 256))

        self.vbox = wx.BoxSizer(wx.VERTICAL)

        # Toolbar
        toolbar = wx.ToolBar(self.frame, -1)
        toolbar.SetToolSeparation(20)
        button = wx.Button(toolbar, -1, 'Add camera', name="buttonAddCam")
        toolbar.AddControl(button)
        toolbar.AddSeparator()
        button = wx.Button(toolbar, -1, 'Save setup', name="buttonSaveSetup")
        toolbar.AddControl(button)
        toolbar.AddSeparator()
        button = wx.Button(toolbar, -1, 'Load setup', name="buttonLoad")
        toolbar.AddControl(button)
        toolbar.AddSeparator()
        button = wx.Button(toolbar, -1, 'Load last setup', name="buttonLoadLast")
        toolbar.AddControl(button)
        toolbar.AddStretchableSpace()
        button = wx.Button(toolbar, -1, 'Tracking calibration', name="buttonTrackingCalib")
        toolbar.AddControl(button)
        toolbar.Realize()
        self.frame.SetToolBar(toolbar)
        self.vbox.Add(toolbar, flag=wx.EXPAND)

        self.box = wx.GridBagSizer(vgap=5, hgap=10)
        self.vbox.Add(self.box, flag=wx.EXPAND)

        # Image view
        self.imageView = ImageViewPanel(self.frame, self.onImageSample)
        self.vbox.Add(self.imageView, flag=wx.EXPAND)

        self.frame.SetSizerAndFit(self.vbox)
        self.frame.Show(True)
#        self.vbox.ComputeFittingWindowSize()

        self.frame.Bind(wx.EVT_BUTTON, self.onButton)
        self.frame.Bind(wx.EVT_TOGGLEBUTTON, self.onToggle)

        return True

    def OnExit(self):
        # val = super(CameraPoseCalibration, self).OnExit()
        for cam in self.cameras:
            cam.destroy()
        return 0

    def onImageSample(self, sample, ogrid):
        # sampleLAB = cv2.cvtColor(sample, cv2.COLOR_RGB2LAB)
        # sampleHSV = cv2.cvtColor(sample, cv2.COLOR_RGB2HSV)
        # print(np.mean(sampleLAB, axis=0), np.mean(sampleLAB, axis=0), np.max(sampleLAB, axis=0))
        # print(np.mean(sampleHSV, axis=0), np.mean(sampleHSV, axis=0), np.max(sampleHSV, axis=0))
        if self.imageViewCamID:
            camera = self.cameras[self.camDict[self.imageViewCamID]]
            sub = camera.getPixels(ogrid)
            sub

#        self.whiteRect = ogrid

    def addCamera(self, imageTopic, cinfoTopic):
        try:
            camera = Camera(self.updateCam)
            camRow = len(self.cameras)
            self.camDict[camera.ID] = camRow
            self.cameras.append(camera)

            pos = wx.GBPosition(camRow, 0)
            flags = wx.EXPAND
            camControls = [(wx.StaticText(self.frame, label=" !", name="textStatus"), 2),
                           (wx.Button(self.frame, label="Image topic", name="buttonImageTopic"), 4),
                           (wx.CheckBox(self.frame, name="checkImageTopic"), 1),
                           (wx.Button(self.frame, label="CameraInfo topic", name="buttonCinfoTopic"), 4),
                           (wx.CheckBox(self.frame, name="checkCinfoTopic"), 1),
                           (wx.Button(self.frame, label="Calibrate", name="buttonCalib"), 2),
                           (wx.Gauge(self.frame, range=CALIB_MAX_TRANSFORM_VECTORS, name="gaugeCalib"), 3),
                           (wx.CheckBox(self.frame, name="checkCalib"), 1),
                           (wx.Button(self.frame, label="Info", name="buttonInfo"), 2),
                           (wx.Button(self.frame, label="View", name="buttonView"), 2),
                           (wx.ToggleButton(self.frame, label="Active", name="toggleActive"), 1)
                           ]

            for (control, cspan) in camControls:
                if type(control) is wx.CheckBox:
                    control.Bind(wx.EVT_LEFT_DOWN, lambda x: None)
                    control.Bind(wx.EVT_LEFT_UP, lambda x: None)
                if control.Name == 'toggleActive':
                    control.SetValue(True)
                camera.__setattr__(control.Name, control.Id)
                self.box.Add(control, pos, span=wx.GBSpan(1, cspan), flag=flags)
                pos.col += cspan
                # print(control, cspan)
            camera.setImageCB(imageTopic)
            camera.setCinfoCB(cinfoTopic)
        except Exception as e:
            print('Unable to add camera: {}'.format(e))
            for i in range(100):
                item = self.box.FindItemAtPosition(wx.GBPosition(camRow, i))
                if item:
                    item.DeleteWindows()
            camera.destroy()
        finally:
            self.vbox.Layout()
            self.frame.Fit()

    def loadCameras(self, fileName):
        with open(os.path.join(self.packagePath, 'data', fileName), 'r') as f:
            try:
                data = rospy.client.yaml.load(f)
                print('Loaded camera(s):', data)
                for cam in data.values():
                    self.addCamera(cam['imageTopic'], cam['cinfoTopic'])
            except IOError:
                wx.LogError("Cannot open file '%s'." % fileName)

    def __plotShow(self, camera):
        if self.imageViewCamID:
            self.__plotHide()
        self.imageViewCamID = camera.ID
        viewButton = self.frame.FindWindowById(camera.buttonView)
        viewButton.SetBackgroundColour(wx.GREEN)
        self.imageView.SetSize(wx.Size(1024, 768))
        self.imageView.Show()
        self.vbox.Layout()
        self.frame.Fit()

    def __plotHide(self):
        self.imageView.Hide()
        camera = self.cameras[self.camDict[self.imageViewCamID]]
        viewButton = self.frame.FindWindowById(camera.buttonView)
        viewButton.SetBackgroundColour(wx.NullColour)
        self.imageViewCamID = None
        self.vbox.Layout()
        self.frame.Fit()

    def __getTopic(self, textBox, listBox):
        if textBox.IsModified():
            topic = textBox.GetLineText(0)
        else:
            try:
                selection = listBox.GetSelection()
            except:  # noqa
                return ''
            if selection == wx.NOT_FOUND:
                return ''
            topic = listBox.GetString(selection)
        return topic

    def onButton(self, event):
        obj = event.GetEventObject()
        flag = wx.EXPAND
        if type(obj.GetParent()) is wx.ToolBar:
            if obj.Name == 'buttonAddCam':
                with wx.Dialog(self.frame, title='Select image and camera info topics') as dlg:
                    box = wx.FlexGridSizer(3, 2, 10, 10)
                    box.SetFlexibleDirection(wx.VERTICAL)
                    # imageList, imageTopicEdit = getTopicListBAndEdit('sensor_msgs/CompressedImage', dlg)
                    imageList, imageTopicEdit = getTopicListBAndEdit('sensor_msgs/Image', dlg)
                    cinfoList, cinfoTopicEdit = getTopicListBAndEdit('sensor_msgs/CameraInfo', dlg)

                    box.Add(imageList, flag=flag)
                    box.Add(cinfoList, flag=flag)

                    box.Add(imageTopicEdit, flag=flag)
                    box.Add(cinfoTopicEdit, flag=flag)

                    cancelButt = wx.Button(dlg, id=wx.ID_CANCEL, label='Cancel')
                    box.Add(cancelButt, flag=flag)
                    okButt = wx.Button(dlg, id=wx.ID_OK, label='OK')
                    box.Add(okButt, flag=flag)

                    dlg.SetSizerAndFit(box)

                    if dlg.ShowModal() == wx.ID_OK:
                        imageTopic = self.__getTopic(imageTopicEdit, imageList)
                        if not imageTopic.strip():
                            print('Invalid image topic')
                            return
                        cinfoTopic = self.__getTopic(cinfoTopicEdit, cinfoList)
                        if not cinfoTopic.strip():
                            print('Invalid camera info topic')
                            return
                    else:
                        print('No topics selected, aborting')
                        return
                rospy.loginfo('Selected image topic: {} and camera info topic: {}'.format(imageTopic, cinfoTopic))
                self.addCamera(imageTopic, cinfoTopic)
            elif obj.Name == 'buttonSaveSetup':
                if len(self.cameras) == 0:
                    return
                d = dict()
                for i, cam in enumerate(self.cameras):
                    d['cam' + str(i)] = {
                        'imageTopic': cam.imageTopic,
                        'cinfoTopic': cam.cinfoTopic}
                dt = datetime.datetime.now()
                fname = dt.strftime('%Y%m%d_%H%M%S') + '.yaml'
                stamp = dt.strftime('%H:%M:%S %d-%m-%Y')
                with open(os.path.join(self.packagePath, 'data', fname), 'w') as f:
                    rospy.client.yaml.dump(d, stream=f, default_flow_style=False)
                last_file = os.path.join(self.packagePath, 'config', 'last_save.yaml')
                try:
                    if not os.path.exists(last_file):
                        f = open(last_file, 'w')
                        f.close()
                    with open(last_file, 'r+') as f:
                        data = rospy.client.yaml.load(f)
                        if data:
                            n = len(data)
                        else:
                            n = 0
                        rospy.client.yaml.dump({'save' + str(n): {'time': stamp, 'file': fname}}, stream=f, default_flow_style=False)
                except IOError:
                    print('Cannot open the "config/last_save.yaml" file')
            elif obj.Name == 'buttonLoad':
                with wx.FileDialog(self.frame, "Select camera file", defaultDir=os.path.join(self.packagePath, 'data'), wildcard="Camera YAML files (*.yaml)|*.yaml", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
                    if fileDialog.ShowModal() == wx.ID_CANCEL:
                        return     # the user changed their mind
                    # Proceed loading the file chosen by the user
                    pathname = fileDialog.GetPath()
                    self.loadCameras(pathname)
            elif obj.Name == 'buttonLoadLast':
                try:
                    lastSave = ''
                    with open(os.path.join(self.packagePath, 'config', 'last_save.yaml'), 'r+') as f:
                        data = rospy.client.yaml.load(f)
                        la = data.values()
                        la.sort(key=lambda d: (datetime.datetime.strptime(d['time'], '%H:%M:%S %d-%m-%Y'), d['file']))
                        lastSave = la[-1]['file']
                    if lastSave:
                        self.loadCameras(lastSave)
                except IOError:
                    print('Cannot open the "config/last_save.yaml" file')
            elif obj.Name == 'buttonTrackingCalib':
                # >>> External calibration <<<
                with wx.Dialog(self.frame, title='Select topic for tracking') as dlg:
                    box = wx.FlexGridSizer(0, 2, 10, 10)
                    box.SetFlexibleDirection(wx.BOTH)
                    topicList, topicEdit = getTopicListBAndEdit(['geometry_msgs/PoseStamped', 'geometry_msgs/TransformStamped', 'geometry_msgs/Pose'], dlg)
                    box.Add(topicList, flag=flag)

                    camList = wx.ListCtrl(dlg, -1, style=wx.LC_REPORT | wx.LC_HRULES | wx.LC_SINGLE_SEL)
                    camList.AppendColumn('Camera', width=250)
                    camList.AppendColumn('Image topic', width=450)
                    camList.AppendColumn('Camera info topic', width=450)
                    for cam in self.cameras:
                        # only show cameras with images and cinfo
                        if cam.phase & (Camera.PHASE_CAM_WAITING_CAM_INFO | Camera.PHASE_CAM_WAITING_IMAGE):
                            continue
                        camList.Append([str(cam.frame_name), cam.imageTopic, cam.cinfoTopic])
                    if camList.GetItemCount() == 0:
                        print('No active cameras, unable to start tracking')
                        return
                    camList.Select(0)
                    box.Add(camList, flag=flag)

                    box.Add(topicEdit, flag=flag)
                    box.AddStretchSpacer()

                    cancelButt = wx.Button(dlg, id=wx.ID_CANCEL, label='Cancel')
                    box.Add(cancelButt, flag=flag)
                    okButt = wx.Button(dlg, id=wx.ID_OK, label='OK')
                    box.Add(okButt, flag=flag)

                    dlg.SetSizerAndFit(box)

                    if dlg.ShowModal() == wx.ID_OK:
                        topic = self.__getTopic(topicEdit, topicList)
                        if not topic.strip():
                            print('Invalid topic')
                            return
                        camera = self.cameras[camList.GetFirstSelected()]
                    else:
                        print('No topics selected, aborting')
                        return
                trackingFrame = TrackerFrame(topic, camera)
                trackingFrame.Center()
                trackingFrame.Show()

        else:
            item = self.box.FindItem(obj)
            if item:
                camRow = item.GetPos().GetRow()
            else:
                print('Invalid item in button callback')
                return

            camera = self.cameras[camRow]

            if obj.Name == 'buttonImageTopic':
                with wx.Dialog(self.frame, title='Select image topic') as dlg:
                    box = wx.FlexGridSizer(4, 1, 10, 10)
                    box.SetFlexibleDirection(wx.VERTICAL)

                    imageList, imageTopicEdit = getTopicListBAndEdit('sensor_msgs/CompressedImage', dlg)
                    box.Add(imageList, flag=flag)
                    box.Add(imageTopicEdit, flag=flag)

                    cancelButt = wx.Button(dlg, id=wx.ID_CANCEL, label='Cancel')
                    box.Add(cancelButt, flag=flag)
                    okButt = wx.Button(dlg, id=wx.ID_OK, label='OK')
                    box.Add(okButt, flag=flag)

                    dlg.SetSizerAndFit(box)
                    if dlg.ShowModal() == wx.ID_OK:
                        imageTopic = self.__getTopic(imageTopicEdit, imageList)
                        if not imageTopic.strip():
                            print('Invalid image topic')
                            return
                    else:
                        print('No topic selected, aborting')
                        return

                rospy.loginfo('Selected image topic: {}'.format(imageTopic))
                toggleActive = self.frame.FindWindowById(camera.toggleActive)
                if not toggleActive.GetValue():
                    toggleActive.SetLabel('Active')
                    toggleActive.SetValue(True)
                camera.setImageCB(imageTopic)
            elif obj.Name == 'buttonCinfoTopic':
                with wx.Dialog(self.frame, title='Select camera info topic') as dlg:
                    box = wx.FlexGridSizer(4, 1, 10, 10)
                    box.SetFlexibleDirection(wx.VERTICAL)

                    cinfoList, cinfoTopicEdit = getTopicListBAndEdit('sensor_msgs/CameraInfo', dlg)
                    box.Add(cinfoList, flag=flag)
                    box.Add(cinfoTopicEdit, flag=flag)

                    cancelButt = wx.Button(dlg, id=wx.ID_CANCEL, label='Cancel')
                    box.Add(cancelButt, flag=flag)
                    okButt = wx.Button(dlg, id=wx.ID_OK, label='OK')
                    box.Add(okButt, flag=flag)

                    dlg.SetSizerAndFit(box)
                    if dlg.ShowModal() == wx.ID_OK:
                        cinfoTopic = self.__getTopic(cinfoTopicEdit, cinfoList)
                        if not cinfoTopic.strip():
                            print('Invalid camera info topic')
                            return
                    else:
                        print('No topic selected, aborting')
                        return

                rospy.loginfo('Selected camera info topic: {}'.format(cinfoTopic))
                camera.setCinfoCB(cinfoTopic)
            elif obj.Name == 'buttonCalib':
                if camera.phase & (Camera.PHASE_CAM_WAITING_CAM_INFO | Camera.PHASE_CAM_WAITING_IMAGE):
                    with wx.MessageDialog(self.frame, 'Camera info or image not received yet!', 'Cannot calibrate camera!', style=wx.OK | wx.CENTER) as dlg:
                        dlg.ShowModal()
                    return
                self.__plotShow(camera)
                camera.calibrate()
            elif obj.Name == 'buttonView':
                if camera.phase & Camera.PHASE_CAM_WAITING_IMAGE:
                    with wx.MessageDialog(self.frame, 'Images not received, yet!', 'No images to show', style=wx.OK | wx.CENTER) as dlg:
                        dlg.ShowModal()
                    return
                if self.imageViewCamID == camera.ID:
                    self.__plotHide()
                else:
                    self.__plotShow(camera)
            elif obj.Name == 'buttonInfo':
                dlg = wx.Dialog(self.frame, -1, 'Camera info "{}"'.format(camera.cinfoTopic))
                box = wx.FlexGridSizer(0, 1, 10, 10)
                box.SetFlexibleDirection(wx.BOTH)
                flags = wx.EXPAND

                textBox = wx.TextCtrl(dlg, -1, style=wx.TE_MULTILINE | wx.TE_READONLY, size=wx.Size(640, 480))
                if hasattr(camera, 'cinfo') and camera.cinfo:
                    cinfo = camera.cinfo
                    textBox.AppendText('frame id = {}\n'.format(cinfo.header.frame_id))
                    textBox.AppendText('Image width = {}\n'.format(cinfo.width))
                    textBox.AppendText('Image height = {}\n'.format(cinfo.height))
                    textBox.AppendText('Distortion parameters = {}\n'.format(cinfo.D))
                    textBox.AppendText('Intrinsic = {}\n'.format(cinfo.K))
                    textBox.AppendText('Projection = {}\n'.format(cinfo.P))
                else:
                    textBox.AppendText('Camera info not received, yet!')
                textBox.SetSize(wx.Size(512, 512))
                box.Add(textBox, flag=flags)
                removeButton = wx.Button(dlg, -1, 'Remove camera', name=str(camRow))
                dlg.Bind(wx.EVT_BUTTON, self.removeCamera, source=removeButton)
                box.Add(removeButton, flag=flags)
                box.Add(wx.Button(dlg, wx.ID_OK, 'Close'), flag=flags)

                dlg.SetSizerAndFit(box)
                dlg.Show()

    def onToggle(self, event):
        obj = event.GetEventObject()
        item = self.box.FindItem(obj)
        if item:
            camRow = item.GetPos().GetRow()
        else:
            print('Invalid item in toggle button callback')
            return
        camera = self.cameras[camRow]
        if obj.Name == 'toggleActive':
            if not obj.GetValue():
                try:
                    camera.imageSubscriber.unregister()
                    rospy.loginfo('Unregisterd from image topic {}'.format(camera.imageTopic))
                    camera.phase |= Camera.PHASE_CAM_WAITING_IMAGE
                except:  # noqa
                    pass  # no one cares
                finally:
                    obj.SetLabel('Inactive')
            else:
                camera.setImageCB(camera.imageTopic)
                obj.SetLabel('Active')
            wx.CallAfter(camera.updateCam_cb, Camera.UPDATE_CHECK_STATUS)

    def removeCamera(self, event):
        obj = event.GetEventObject()
        camRow = int(obj.Name)
        camera = self.cameras[camRow]
        with wx.MessageDialog(self.frame, 'You are about to remove a camera tracker!\nCam: {}'.format(camera.frame_name), 'Remove camera', style=wx.OK | wx.CANCEL | wx.CENTER) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            for i in range(100):
                item = self.box.FindItemAtPosition(wx.GBPosition(camRow, i))
                if item:
                    item.DeleteWindows()
        self.box.Layout()
        self.frame.Fit()
        camera.destroy()
        obj.GetParent().Close()

    def updateCam(self, camera, flag=0, img=None):
        if (img is not None) and self.imageViewCamID == camera.ID:
            wx.CallAfter(self.updateImage, img)

        if flag & Camera.UPDATE_CHECK_PROGRESS:
            if camera.phase & Camera.PHASE_CAM_EXT_CALIB:
                try:
                    calibGauge = self.frame.FindWindowById(camera.gaugeCalib)
                    gpos = len(camera.rvecsBuffer)
                    val = gpos if gpos <= calibGauge.GetRange() else calibGauge.GetRange()
                    calibGauge.SetValue(val)
                except:  # noqa
                    pass  # just ignore, 'cause I'm tired of this shit

        if flag & Camera.UPDATE_CHECK_STATUS:
            checkCinfoTopic = self.frame.FindWindowById(camera.checkCinfoTopic)
            if camera.phase & Camera.PHASE_CAM_WAITING_CAM_INFO:
                checkCinfoTopic.SetValue(False)
            else:
                checkCinfoTopic.SetValue(True)
            checkImageTopic = self.frame.FindWindowById(camera.checkImageTopic)
            if camera.phase & Camera.PHASE_CAM_WAITING_IMAGE:
                checkImageTopic.SetValue(False)
            else:
                checkImageTopic.SetValue(True)
            checkCalib = self.frame.FindWindowById(camera.checkCalib)
            checkCalib.SetValue(camera.isPoseCalibrated)

            textStatus = self.frame.FindWindowById(camera.textStatus)
            if checkCinfoTopic.GetValue() and checkImageTopic.GetValue() and checkCalib.GetValue():
                textStatus.SetLabel('')
            else:
                textStatus.SetLabel(' !')

    def updateImage(self, img):
        if (len(img.shape) == 3) and (img.shape[2] == 3):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.whiteRect:
            # print(self.whiteRect)
            cv2.rectangle(img, tuple(self.whiteRect.topLeft), tuple(self.whiteRect.bottomRight), (0, 255, 0))

        wx.CallAfter(self.imageView.update, img)


def main(args):
    pe = CameraPoseCalibration()
    pe.MainLoop()
    print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
