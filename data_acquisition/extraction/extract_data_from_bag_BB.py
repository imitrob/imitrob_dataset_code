# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract images and depth data from a rosbag.
"""
import os
import sys
from geometry_msgs.msg import TransformStamped

from tf_bag import BagTfTransformer
import argparse
import json
import os
import rosbag
import rospy
import cv2
import tf
from cv_bridge import CvBridge
import numpy as np
from tf.transformations import quaternion_matrix, quaternion_from_matrix, translation_from_matrix
import tf
from numpy import genfromtxt
from scipy.spatial import ConvexHull, Delaunay


class myBagTransformer(object):
    def __init__(self, path, bagName):
        self.bag = rosbag.Bag(os.path.join(path, bagName), "r")
        self.bag_transformer = BagTfTransformer(self.bag)

    def lookupTransform(self, orig_frame, dest_frame, t):
        trans, rot = self.bag_transformer.lookupTransform(orig_frame=orig_frame, dest_frame=dest_frame,
                                                          time=t)
        return trans, rot

    def world_to_board (self):
        #provides transform between world_vive to board
        trans = np.empty([1,3],dtype = float)
        rot = np.empty([1,4],dtype = float)
        count = 0
        for topic, msg, t in self.bag.read_messages(topics=['/tf']):
            if count != 1:
                if msg.transforms[0].header.frame_id == args.world_frame:
                    if msg.transforms[0].child_frame_id == args.board_frame:
                        print(msg)
                        trans = np.array([msg.transforms[0].transform.translation.x,
                                           msg.transforms[0].transform.translation.y,
                                           msg.transforms[0].transform.translation.z])
                        rot = np.array([msg.transforms[0].transform.rotation.x, msg.transforms[0].transform.rotation.y,
                                        msg.transforms[0].transform.rotation.z, msg.transforms[0].transform.rotation.w])
                        count = 1
            else:
                break
        return trans, rot

    def world_to_camera(self,topicC):#topic "camera1_color_optical_frame", "camera2_color_optical_frame"
        #provides transform between world_vive to camera (using world_to_board transform)
        count1 = 0
        Ctran = np.empty([1,3],dtype = float)
        Crot = np.empty([1,4],dtype = float)
        for topic, msg, t in self.bag.read_messages(topics=['/tf']):
            if count1 != 1:
                if msg.transforms[0].header.frame_id == args.board_frame:
                    if msg.transforms[0].child_frame_id == topicC:
                        print(msg)
                        Ctran = np.array([msg.transforms[0].transform.translation.x,
                                            msg.transforms[0].transform.translation.y,
                                            msg.transforms[0].transform.translation.z])
                        Crot = np.array([msg.transforms[0].transform.rotation.x, msg.transforms[0].transform.rotation.y,
                                         msg.transforms[0].transform.rotation.z, msg.transforms[0].transform.rotation.w])
                        count1 = 1
            else:
                break

        #getting transform from world_vive to board
        Btran, Brot = self.world_to_board()
        Rt_world_to_board = self.make_Rt_matrix(Btran,Brot)

        #getting transform from board to camera
        Rt_board_to_camera = self.make_Rt_matrix(Ctran,Crot)

        #getting transform from world_vive to camera
        Rt_world_to_camera = np.matmul(Rt_world_to_board,Rt_board_to_camera)
        return Rt_world_to_camera

    def transform_from_to(self, from_topic, to_topic):#topic "camera1_color_optical_frame", "camera2_color_optical_frame"
        #provides tranform between from_topic parent frame to to_topic (child_frame) - only between neigboring nodes
        count1 = 0
        trans = np.empty([1,3],dtype = float)
        rot = np.empty([1,4],dtype = float)
        for topic, msg, t in self.bag.read_messages(topics=['/tf']):
            if count1 != 1:
                if msg.transforms[0].header.frame_id == from_topic:
                    if msg.transforms[0].child_frame_id == to_topic:
                        #print(msg)
                        trans = np.array([msg.transforms[0].transform.translation.x,
                                            msg.transforms[0].transform.translation.y,
                                            msg.transforms[0].transform.translation.z])
                        rot = np.array([msg.transforms[0].transform.rotation.x, msg.transforms[0].transform.rotation.y,
                                         msg.transforms[0].transform.rotation.z, msg.transforms[0].transform.rotation.w])
                        count1 = 1
            else:
                break
        return trans, rot

    def make_Rt_matrix(self, trans, rot):
        #gives back rotation_translation matrix 4x4 from translation and rotation vector
        Rt_mat = quaternion_matrix(rot)
        Rt_mat[0:3,3] = trans
        return Rt_mat


def highlight_bbox(img, bbox_array):
    bbox_hull = ConvexHull(bbox_array)
    bbox_hull_points = bbox_hull.points[bbox_hull.vertices, :]
    bbox_dhp = Delaunay(bbox_hull_points)

    h = np.shape(img)[0]
    w = np.shape(img)[1]
    points_test = np.transpose(np.meshgrid(range(h), range(w)))
    points_test = np.reshape(points_test, [h * w, 2])
    points_inside_hull = np.asarray(bbox_dhp.find_simplex(points_test) >= 0)
    mask = np.reshape(points_inside_hull, [h, w])

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img[mask == 0, :] = 0 * img[mask == 0, :] + 1 * np.dstack([img_gray] * 3)[mask == 0, :]
    return img


def get_pose_interpolator(messages, source, target, max_interpolation_time=0.1):
    tf_msgs = []
    for msg in messages:
        for transform in msg.transforms:  # type: TransformStamped
            # if not isinstance(transform, TransformStamped):
            #     continue
            assert isinstance(transform, TransformStamped)
            if transform.header.frame_id == source and transform.child_frame_id == target:
                tf_msgs.append(transform)

    points = np.zeros((len(tf_msgs), 7))

    for idx, msg in enumerate(tf_msgs):
        points[idx, 3:6] = tf.transformations.euler_from_quaternion([msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z,msg.transform.rotation.w], axes='sxyz')
        points[idx, :3] = np.array([msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z])
        points[idx, 6] = msg.header.stamp.to_sec()

    def interpolator(t):
        if t < points[0,6] or t > points[-1,6]:
            raise ValueError('Requested time would need extrapolation.')
        smaller = np.where(points[:,6] - t < 0.0)
        larger = np.where(points[:, 6] - t >= 0.0)


        p1 = points[np.max(smaller)]
        p2 = points[np.min(larger)]

        if (p2[6] - p1[6]) >= max_interpolation_time:
            raise ValueError('Data not dense enough at requested time.')

        p = p1[:6] + (t - p1[6])/(p2[6] - p1[6]) * (p2[:6] - p1[:6])

        return p

    return interpolator


def main(args):
    inputNames = [(bag, calib) for bag, calib in zip(args.input[::2], args.input[1::2])]
    camera1_color_topic = args.camera1_color
    camera1_detph_topic = args.camera1_detph
    camera2_color_topic = args.camera2_color
    camera2_detph_topic = args.camera2_detph

    diff_threshold = 0.02
    try:  # solving compatibility issues between ROS versions:
        diff_threshold = rospy.Duration(diff_threshold)
    except:
        diff_threshold = rospy.Duration(nsecs=int(diff_threshold * 1e9))
    colors = [[  0,255,  0,255,  0,255,  0,255,127],
              [  0,  0,255,255,  0,  0,255,255,127],
              [  0,  0,  0,  0,255,255,255,255,127]]
    borders = [0, 0, 0, 0, 0, 0, 0, 0]
    bbox_radius = 6

    tools = ('gluegun', 'grout','hammer','roller') # set of tools, name of the tool should be in the file name (if new one used, add to the end of the list)
    subjects = ('R','J','K','G') # set of subjects - if there is a new one, add to the end, name of the subject should be in the file name

    for bagName, calibFile in inputNames:
        path, bagName = os.path.split(bagName)

        bagfName = ""
        _K1 = np.eye(3)
        _K2 = np.eye(3)
        subjectSel = 'None'
        subjectID = 0
        toolSel = 'None'

        for i, tool in enumerate(tools):
            pos = bagName.find(tool)
            if pos > 0:
                toolSel = tool

        for i, subject in enumerate(subjects):
            pos = bagName.find(subject)
            if pos > 0:
                subjectSel = subject
                subjectID = i

        background = 'green' if 'green' in bagName else 'normal'

        bag = rosbag.Bag(os.path.join(path, bagName), "r")

        mbt = myBagTransformer(path, bagName)
        RtWorldToCam1 = mbt.world_to_camera(args.camera1_optical)
        RtWorldToCam2 = mbt.world_to_camera(args.camera2_optical)

        print('RtWorldToCam1', RtWorldToCam1)
        print('RtWorldToCam2', RtWorldToCam2)

        # pose interpolation function for htc vive poses
        msg = [msg for topic, msg, t in bag.read_messages('/tf')]
        pose_interp = get_pose_interpolator(msg, args.world_frame, args.tracker_frame)


        bridge = CvBridge()
        count = 0
        output_folder = os.path.join(path, bagName[0:-4])

        for _, msg, _ in bag.read_messages(args.camera1_info):
            _K1 = np.array(msg.K).reshape((3, 3))
            break
        print(f"C1 camera matrix:\n{_K1}")

        for _, msg, _ in bag.read_messages(args.camera2_info):
            _K2 = np.array(msg.K).reshape((3, 3))
            break
        print(f"C2 camera matrix:\n{_K2}")

        try:  # else already exist
            os.makedirs(output_folder + "/Image")
            os.makedirs(output_folder + "/Depth")
            os.makedirs(output_folder + "/6DOF")
            os.makedirs(output_folder + "/BBox")
            os.makedirs(output_folder + "/BBox_visualization")
        except:
            pass

        x = []
        gend = bag.read_messages(camera1_detph_topic)
        gend2 = bag.read_messages(camera2_detph_topic)
        genI2 = bag.read_messages(camera2_color_topic)
        d = list(gend)
        d = np.array(d)
        i2 = list(genI2)
        i2 = np.array(i2)
        d2 = list(gend2)
        d2 = np.array(d2)
        dtime = np.array([rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs) for topic, msg, t in
                          bag.read_messages(camera1_detph_topic)])
        d2time = np.array([rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs) for topic, msg, t in
                          bag.read_messages(camera2_detph_topic)])
        i2time = np.array([rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs) for topic, msg, t in
                          bag.read_messages(camera2_color_topic)])

        print("Found {} timestamps in the bag.".format(len(dtime)))

        for _, msg, t in bag.read_messages(topics=camera1_color_topic):
            refTime=rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
            messageIndexD = np.argmin(np.abs(dtime - rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)))
            diff = abs(dtime[messageIndexD] - rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs))
            messageIndexD2 = np.argmin(np.abs(d2time - rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)))
            diff2 = abs(d2time[messageIndexD2] - rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs))
            messageIndexI2 = np.argmin(np.abs(i2time - rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)))
            diff3 = abs(i2time[messageIndexI2] - rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs))

            if diff < diff_threshold and diff2 < diff_threshold and diff3 < diff_threshold:
                # image data saving
                cv_image1 = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")
                fname1 = bagfName + 'C1' + "F%04i.png" % count
                cv2.imwrite(os.path.join(output_folder + "/Image/", fname1), cv_image1)

                img2_msg = i2[messageIndexI2][1]
                cv_image2 = bridge.compressed_imgmsg_to_cv2(img2_msg, desired_encoding="passthrough")
                fname2 = bagfName + 'C2' + "F%04i.png" % count
                cv2.imwrite(os.path.join(output_folder + "/Image/", fname2), cv_image2)

                # depth data saving
                depth_msg = d[messageIndexD, 1]
                cv_imaged1 = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                depth_array = np.array(cv_imaged1, dtype=np.float32)
                depth = np.array(depth_array, dtype=np.uint16)
                cv2.imwrite(os.path.join(output_folder + "/Depth/", bagfName + 'C1' + "F%04i.png" % count), depth)

                depth2_msg = d2[messageIndexD2, 1]
                cv_imaged2 = bridge.imgmsg_to_cv2(depth2_msg, desired_encoding="passthrough")
                depth2_array = np.array(cv_imaged2, dtype=np.float32)
                depth2 = np.array(depth2_array, dtype=np.uint16)
                cv2.imwrite(os.path.join(output_folder + "/Depth/", bagfName + 'C2' + "F%04i.png" % count),depth2)

                # 6DOF saving
                common_time1 = refTime - rospy.Duration.from_sec(0.17)
                try:
                    p = pose_interp(common_time1.to_sec())
                except ValueError as e:
                    print(e)
                    # print('Skipping image')
                    print("Skipped image %i" % count)
                    count += 1
                    continue

                world_to_tracker2_translation = p[:3]
                world_to_tracker2_quaternion = tf.transformations.quaternion_from_euler(p[3], p[4], p[5], axes='sxyz')

                frame = "F%04i" % count

                data = {
                    frame: {
                        "Image_C1": {
                            "stamp": {"sec": msg.header.stamp.secs,
                                        "nsecs": msg.header.stamp.nsecs}
                        },
                        "Image_C2": {
                            "stamp": {"sec": img2_msg.header.stamp.secs,
                                        "nsecs": img2_msg.header.stamp.nsecs}
                        },
                        "Depth_C1": {
                            "stamp": {"sec": depth_msg.header.stamp.secs,
                                        "nsec": depth_msg.header.stamp.nsecs}
                        },
                        "Depth_C2": {
                            "stamp": {"sec": depth2_msg.header.stamp.secs,
                                        "nsec": depth2_msg.header.stamp.nsecs}
                        },

                        "6DOF": {
                            "stamp": {"sec": common_time1.secs,
                                        "nsec": common_time1.nsecs},
                            "translation": {"x": world_to_tracker2_translation[0],
                                            "y": world_to_tracker2_translation[1],
                                            "z": world_to_tracker2_translation[2]},
                            "rotation": {
                                "x": world_to_tracker2_quaternion[0],
                                "y": world_to_tracker2_quaternion[1],
                                "z": world_to_tracker2_quaternion[2],
                                "w": world_to_tracker2_quaternion[3]
                            }
                        }
                    }
                }

                fpath = os.path.join(output_folder + "/6DOF/", bagfName + "F%04i.json" % count)
                with open(fpath, "w") as write_file:
                    json.dump(data, write_file, indent=4, sort_keys=True)

                # save common data in common.json for each bagfile
                # cannonical coordinates of BB in tracker coordinates
                BB_tracker = genfromtxt(calibFile, delimiter=',')

                dataCommon = {
                    "K_C1": np.array(_K1.reshape(1, 9)).tolist(),
                    "K_C2": np.array(_K2.reshape(1, 9)).tolist(),
                    "tool": toolSel,
                    "calibration_file": calibFile,
                    "bagfName": bagName,
                    "subject_ID": subjectID,
                    "subject_name": subjectSel,
                    "background_type": background,
                    "BB_in_tracker": {
                        'V1': {'x': BB_tracker[0][0], 'y': BB_tracker[0][1], 'z': BB_tracker[0][2]},
                        'V2': {'x': BB_tracker[1][0], 'y': BB_tracker[1][1], 'z': BB_tracker[1][2]},
                        'V3': {'x': BB_tracker[2][0], 'y': BB_tracker[2][1], 'z': BB_tracker[2][2]},
                        'V4': {'x': BB_tracker[3][0], 'y': BB_tracker[3][1], 'z': BB_tracker[3][2]},
                        'V5': {'x': BB_tracker[4][0], 'y': BB_tracker[4][1], 'z': BB_tracker[4][2]},
                        'V6': {'x': BB_tracker[5][0], 'y': BB_tracker[5][1], 'z': BB_tracker[5][2]},
                        'V7': {'x': BB_tracker[6][0], 'y': BB_tracker[6][1], 'z': BB_tracker[6][2]},
                        'V8': {'x': BB_tracker[7][0], 'y': BB_tracker[7][1], 'z': BB_tracker[7][2]}
                    }
                }

                fpath = os.path.join(output_folder + "/parameters.json")

                with open(fpath, "w") as write_file:
                    json.dump(dataCommon, write_file, indent=4, sort_keys=True)

                # for each frame, transform and project the BB to camera frames C1, C2
                tracker_to_camera1_translation = []

                try:
                    common_time = refTime - rospy.Duration.from_sec(0.17)

                    common_time = refTime  - rospy.Duration.from_sec(0.17)
                    try:
                        p = pose_interp(common_time.to_sec())
                    except ValueError as e:
                        print(e)
                        print("Skipped image %i" % count)

                        count += 1
                        continue

                    world_to_tracker2_translation = p[:3]
                    world_to_tracker2_quaternion = tf.transformations.quaternion_from_euler(p[3], p[4], p[5],
                                                                                            axes='sxyz')

                    RtWorldToTracker = mbt.make_Rt_matrix(world_to_tracker2_translation, world_to_tracker2_quaternion)
                    RtTrackerToWorld = np.linalg.inv(RtWorldToTracker)
                    RtTrackerToCamera1 = np.dot(RtTrackerToWorld,RtWorldToCam1)
                    RtTrackerToCamera2 = np.dot(RtTrackerToWorld, RtWorldToCam2)
                    tracker_to_camera1_quaternion = quaternion_from_matrix(RtTrackerToCamera1)
                    tracker_to_camera1_translation = translation_from_matrix(RtTrackerToCamera1)
                    tracker_to_camera2_quaternion = quaternion_from_matrix(RtTrackerToCamera2)
                    tracker_to_camera2_translation = translation_from_matrix(RtTrackerToCamera2)


                    # Goes through the corners and first projects them to the camera frame, then uses pinhole camera model to project them to images
                    corners = BB_tracker[:]
                    CornersCam1 = np.zeros((8, 3))
                    Corners2DC1 = np.zeros((8, 2))
                    CornersCam2 = np.zeros((8, 3))
                    Corners2DC2 = np.zeros((8, 2))
                    for i, corner in enumerate(corners):
                        pointC1 = np.dot(np.linalg.inv(RtTrackerToCamera1), np.hstack((corner,1)).T)
                        CornersCam1[i, :] = np.dot(_K1, pointC1[:3])
                        Corners2DC1[i, 0] = CornersCam1[i, 0] / CornersCam1[i, 2]
                        Corners2DC1[i, 1] = CornersCam1[i, 1] / CornersCam1[i, 2]

                        pointC2 = np.dot(np.linalg.inv(RtTrackerToCamera2), np.hstack((corner,1)).T)
                        CornersCam2[i, :] = np.dot(_K2, pointC2[:3])
                        Corners2DC2[i, 0] = CornersCam2[i, 0] / CornersCam2[i, 2]
                        Corners2DC2[i, 1] = CornersCam2[i, 1] / CornersCam2[i, 2]

                    dataBB = {
                        frame: {
                            "tracker_to_C1": {
                                "stamp": {"sec": common_time.secs,
                                        "nsec": common_time.nsecs},
                                "translation": {"x": tracker_to_camera1_translation[0],
                                                "y": tracker_to_camera1_translation[1],
                                                "z": tracker_to_camera1_translation[2]},
                                "rotation": {
                                    "x": tracker_to_camera1_quaternion[0],
                                    "y": tracker_to_camera1_quaternion[1],
                                    "z": tracker_to_camera1_quaternion[2],
                                    "w": tracker_to_camera1_quaternion[3]
                                }
                            },
                            "tracker_to_C2": {
                                "stamp": {"sec": common_time.secs,
                                        "nsec": common_time.nsecs},
                                "translation": {"x": tracker_to_camera2_translation[0],
                                                "y": tracker_to_camera2_translation[1],
                                                "z": tracker_to_camera2_translation[2]},
                                "rotation": {
                                    "x": tracker_to_camera2_quaternion[0],
                                    "y": tracker_to_camera2_quaternion[1],
                                    "z": tracker_to_camera2_quaternion[2],
                                    "w": tracker_to_camera2_quaternion[3]
                                }
                            },
                            "BBox_2D_C1": {
                                'V1': {'u': Corners2DC1[0][0], 'v': Corners2DC1[0][1]},
                                'V2': {'u': Corners2DC1[1][0], 'v': Corners2DC1[1][1]},
                                'V3': {'u': Corners2DC1[2][0], 'v': Corners2DC1[2][1]},
                                'V4': {'u': Corners2DC1[3][0], 'v': Corners2DC1[3][1]},
                                'V5': {'u': Corners2DC1[4][0], 'v': Corners2DC1[4][1]},
                                'V6': {'u': Corners2DC1[5][0], 'v': Corners2DC1[5][1]},
                                'V7': {'u': Corners2DC1[6][0], 'v': Corners2DC1[6][1]},
                                'V8': {'u': Corners2DC1[7][0], 'v': Corners2DC1[7][1]}
                            },
                            "BBox_2D_C2": {
                                'V1': {'u': Corners2DC2[0][0], 'v': Corners2DC2[0][1]},
                                'V2': {'u': Corners2DC2[1][0], 'v': Corners2DC2[1][1]},
                                'V3': {'u': Corners2DC2[2][0], 'v': Corners2DC2[2][1]},
                                'V4': {'u': Corners2DC2[3][0], 'v': Corners2DC2[3][1]},
                                'V5': {'u': Corners2DC2[4][0], 'v': Corners2DC2[4][1]},
                                'V6': {'u': Corners2DC2[5][0], 'v': Corners2DC2[5][1]},
                                'V7': {'u': Corners2DC2[6][0], 'v': Corners2DC2[6][1]},
                                'V8': {'u': Corners2DC2[7][0], 'v': Corners2DC2[7][1]}
                            }
                        }
                    }

                    fpath = os.path.join(output_folder + "/BBox/", bagfName + "F%04i.json" % count)
                    with open(fpath, "w") as write_file:
                        json.dump(dataBB, write_file, indent=4, sort_keys=True)

                    for cc in [1, 2]:
                        camera = "C{}".format(cc)  # e.g. 'C1'
                        bbox_dict = dataBB[frame]["BBox_2D_{}".format(camera)]
                        bbox_array = [[vert["v"], vert["u"]] for k, vert in bbox_dict.items()]
                        bbox_array = np.asarray(np.round(bbox_array), int)
                        img = eval("cv_image{}".format(cc))[:, :, ::-1]
                        img = highlight_bbox(img, bbox_array)
                        h, w = np.shape(img)[:2]
                        for bv in range(8):
                            y, x = bbox_array[bv, :2]
                            if -bbox_radius <= x < w + bbox_radius and -bbox_radius <= y < h + bbox_radius:
                                for k in range(3):
                                    img[max(y-bbox_radius, 0):min(y+bbox_radius+1, h), max(x-bbox_radius, 0):min(x+bbox_radius+1, w), k] = borders[bv]
                                    img[max(y-bbox_radius+1, 0):min(y+bbox_radius, h), max(x-bbox_radius+1, 0):min(x+bbox_radius, w), k] = colors[k][bv]

                        fname_output = os.path.join(output_folder, 'BBox_visualization', eval("fname{}".format(cc)))
                        cv2.imwrite(fname_output, img[:, :, ::-1])

                except Exception as e:
                    print(e)

                print ("Wrote image %i" % count)
            else:
                print( "Skipped image %i" % count)

            count += 1
        bag.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(epilog='Example:\n$ python extract_data_from_bag_BB.py path_to_data/<tool>_<subject>_<green_or_task>_<hand>_<camera>.bag <calibration_path>/<tool_calibration_file>.csv')
    parser.add_argument('input', nargs=argparse.REMAINDER, type=str)
    parser.add_argument('--camera1-color', '--c1', type=str, default="/camera1/color/image_raw/compressed", help="ROS topic containing the color images of the primary camera (c1 color).")
    parser.add_argument('--camera1-depth', '--d1', type=str, default='/camera1/depth/image_rect_raw', help="ROS topic containing the depth images of the primary camera (c1 depth).")
    parser.add_argument('--camera1-info', '--i1', type=str, default='/camera1/color/camera_info', help="ROS topic containing the camera info of the primary camera (c1 info).")
    parser.add_argument('--camera1-optical', '--o1', type=str, default='camera1_color_optical_frame', help="ROS (coordinate) frame name of the primary camera.")

    parser.add_argument('--camera2-color', '--c2', type=str, default="/camera2/color/image_raw/compressed", help="ROS topic containing the color images of the secondary camera (c2 color).")
    parser.add_argument('--camera2-depth', '--d2', type=str, default='/camera2/depth/image_rect_raw', help="ROS topic containing the depth images of the secondary camera (c2 depth).")
    parser.add_argument('--camera2-info', '--i2', type=str, default='/camera2/color/camera_info', help="ROS topic containing the camera info of the secondary camera (c2 info).")
    parser.add_argument('--camera2-optical', '--o2', type=str, default='camera2_color_optical_frame', help="ROS (coordinate) frame name of the secondary camera.")

    parser.add_argument('--world-frame', '--wf', type=str, default='world_vive', help="ROS (coordinate) frame of the HTC Vive (main frame).")
    parser.add_argument('--tracker-frame', '--tf', type=str, default="tracker_LHR_786752BF", help="ROS (coordinate) frame of the HTC Vive (main frame).")
    parser.add_argument('--board-frame', '--bf', type=str, default='board', help="ROS (coordinate) frame of the calibration checker board.")

    args = parser.parse_args()
    input_nargs = len(args.input)
    if input_nargs > 0:
        if input_nargs % 2 != 0:
            print("Incorrect number of input arguments! The input paths should always come in pairs of 'bagfile_path, calibration_file_path'.")
            sys.exit(-1)
    else:
        print("No input paths specified! Please specify a path to at least one bagfile and a calibration file.")
        sys.exit(-1)
    main(args)

    """Example usege:

    $ python extract_data_from_bag_BB.py path_to_data/gluegun_G_green_R_c2_bag_2021-10-19-14-26-22.bag calibration_path/gluegun_trace_bag_2021-10-19-11-07-51_bb_corners_z_aligned.csv
    """
