#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:17:48 2019

@author: matusmacbookpro
"""

import os
import re
import numpy as np
import csv
#import tf


def alphanum_key(s):
#    import re

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    return [tryint(c) for c in re.split('([0-9]+)', s)]


def get_list_of_files(path, sort_alphanum=True, extension_list=[]):
#    import os
    if len(extension_list) == 0:
        fname_list = [fname for fname in os.listdir(path) if os.path.isfile(os.path.join(path, fname))]
    else:
        fname_list = []
        for extension in extension_list:
            fname_list.append([fname for fname in os.listdir(path) if os.path.isfile(os.path.join(path, fname)) and fname.endswith('.' + extension)])
    if sort_alphanum:
        fname_list.sort(key=alphanum_key)
    return fname_list


def get_list_of_files_from_file(path, fname='selected_files.txt', extension=''):
    with open(os.path.join(path, fname), 'r') as f:
        fname_list = f.read().splitlines()
    if len(extension) > 0:
        if extension[0] != '.':
            extension = '.' + extension
        for i in range(len(fname_list)):
            fname_list[i] = fname_list[i] + extension
    return fname_list


def extract_tra_rotquat(fname):
    # Extract translation (xyz vector) and rotation (quat_xyzw quaternion vector)
    # From .txt file with structure:
    # 0    filename.png
    # ...
    # -7    0.686005592346
    # -6    -0.908927559853
    # -5    -2.62150764465
    # -4    0.814501161889
    # -3    0.0823923594018
    # -2    0.494164365842
    # -1    -0.292576376226
    with open(fname) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]  # remove whitespaces
    while len(lines[-1]) == 0:  # remove empty trailing lines
        lines = lines[:-1]
    xyz = [float(x) for x in lines[-7:-4]]
    quat_xyzw = [float(x) for x in lines[-4:]]
    return xyz, quat_xyzw


def extract_timestamps(fname):
    # Extract timestamps of image and corresponding pose
    # From .txt file with structure:
    # 0   filename.png
    # 1   519826934
    # 2   153877573
    # 3   1519826934
    # 4   335010485
    # -9   1519826934
    # -8   150369406
    # ...
#    import numpy as np
    with open(fname) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]  # remove whitespaces
    while len(lines[-1]) == 0:  # remove trailing empty lines
        lines = lines[:-1]
    time_rgb = np.float64(lines[1] + '.' + lines[2].zfill(9))
#    time_depth = np.float64(lines[3] + '.' + lines[4].zfill(9))
    time_pose = np.float64(lines[-9] + '.' + lines[-8].zfill(9))
    return time_rgb, time_pose


def get_tra_rot_from_row(row, header, old_csv=True):
    index_translation = header.index('translation')
    index_rotation = header.index('rotation')
    if old_csv:
        index_translation -= 1
        index_rotation -= 1
    index_tra_x = index_translation + 1
    index_tra_y = index_tra_x + 1
    index_tra_z = index_tra_y + 1
    index_rot_x = index_rotation + 1
    index_rot_y = index_rot_x + 1
    index_rot_z = index_rot_y + 1
    index_rot_w = index_rot_z + 1
    tra_x = row[index_tra_x]
    tra_y = row[index_tra_y]
    tra_z = row[index_tra_z]
    rot_x = row[index_rot_x]
    rot_y = row[index_rot_y]
    rot_z = row[index_rot_z]
    rot_w = row[index_rot_w]
    tra = [tra_x, tra_y, tra_z]
    rot = [rot_x, rot_y, rot_z, rot_w]
    return tra, rot


def get_parent_child_from_row(row, header, old_csv=True):
    index_parent = header.index('frame_id')
    index_child = header.index('child_frame_id')
    if old_csv:
        index_parent -= 1
        index_child -= 1
    parent = row[index_parent].strip('"')
    child = row[index_child].strip('"')
    return parent, child


def get_transformation_for_parent_child(row_list, header, parent='world_vive', child='board'):
    tra = None
    rot = None
    for row in row_list:
        parent_row, child_row = get_parent_child_from_row(row, header)
        if parent_row == parent and child_row == child:
            tra, rot = get_tra_rot_from_row(row, header)
            break
    return tra, rot


def get_transformations_from_csv(fname_csv, parent_list=['world_vive', 'board'],
                                 child_list=['board', 'camera_rgb_optical_frame']):
    """
    Get transformations from CSV files
    - tra = x, y, z
    - rot = x, y, z, w
    :param fname_csv:
    :param parent_list:
    :param child_list:
    :return:
    """
#    import csv
    with open(fname_csv, 'r') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        header = next(reader)
        row_list = [row for row in reader]

    tra_list = []
    rot_list = []
    for i in range(len(parent_list)):
        parent = parent_list[i]
        child = child_list[i]
        tra, rot = get_transformation_for_parent_child(row_list=row_list, header=header, parent=parent, child=child)
        tra_list.append(tra)
        rot_list.append(rot)
    return tra_list, rot_list


def get_board2world_cam2board_from_csv(fname_csv, camera_name='camera_rgb_optical_frame'):
    tra_list, rot_list = get_transformations_from_csv(fname_csv, parent_list=['world_vive', 'board'],
                                                      child_list=['board', camera_name])
    R_board2world = rot_list[0]
    t_board2world = tra_list[0]
    R_cam2board = rot_list[1]
    t_cam2board = tra_list[1]
    return R_board2world, t_board2world, R_cam2board, t_cam2board


def Rt_end2cam(R_controller2world, t_controller2world, R_board2world, t_board2world, R_cam2board, t_cam2board):
    """
    Calculate Rt matrix (rotation & translation) from endpoint to camera
    :param R_controller2world:
    :param t_controller2world:
    :param R_board2world:
    :param t_board2world:
    :param R_cam2board:
    :param t_cam2board:
    :return:
    """
#    import numpy as np
    import tf

    R_controller2world = tf.transformations.quaternion_matrix(R_controller2world)
    t_controller2world = tf.transformations.translation_matrix(t_controller2world)

    # constant transformation
    t_end2controller = np.array([0.00481989,  0.04118873, 0.21428937, 1]).reshape(4, 1)
    R_0 = tf.transformations.euler_matrix(0, -np.pi / 2, 0)

    t_board2world = tf.transformations.translation_matrix(t_board2world)
    R_board2world = tf.transformations.quaternion_matrix(R_board2world)

    t_cam2board = tf.transformations.translation_matrix(t_cam2board)
    R_cam2board = tf.transformations.quaternion_matrix(R_cam2board)

    Rt_controller2world = tf.transformations.concatenate_matrices(t_controller2world, R_controller2world)
    t_end2world = tf.transformations.translation_matrix(Rt_controller2world.dot(t_end2controller)[:3, :].reshape(3, ))
    R_end2world = R_controller2world.dot(R_0)
    Rt_end2world = tf.transformations.concatenate_matrices(t_end2world, R_end2world)
    Rt_board2world = tf.transformations.concatenate_matrices(t_board2world, R_board2world)
    Rt_world2board = np.linalg.inv(Rt_board2world)
    Rt_cam2board = tf.transformations.concatenate_matrices(t_cam2board, R_cam2board)
    Rt_board2cam = np.linalg.inv(Rt_cam2board)
    Rt_end2cam = tf.transformations.concatenate_matrices(Rt_board2cam, Rt_world2board, Rt_end2world)
    return Rt_end2cam


def generate_Rt_matrices(path, fname_csv, folder_input='6DOF', folder_output='pose',
                         camera_name='camera_rgb_optical_frame', selected_files_only=True,fname='selected_files.txt'):
    """
    Generate Rt matrices from 6DOF pose files
    :param path: Folder with data
    :param fname_csv: Name of csv file
    :param folder_input: Subfolder with 6DOF pose files
    :param folder_output: Subfolder with Rt pose files
    :param camera_name: Name of camera TODO
    :param selected_files_only: Use only selected files
    :return: number of generated Rt matrices
    """

#    import os
#    import numpy as np
    
    print('\n=== Generating Rt matrices ===')

    path_input = os.path.join(path, folder_input)
    path_output = os.path.join(path, folder_output)
    if not os.path.isdir(path_output):
        os.makedirs(path_output)
    if selected_files_only:
        
        fname_input_list = []
        
        all_fname_input_list = get_list_of_files_from_file(path=path,fname=fname, extension='.txt')
        
        data_type = fname_csv.split('.')[0]
#        last two letters (TF) is not in the image file name
        data_type = data_type[0:-2]
        
        for fn in all_fname_input_list:
            
            if data_type in fn:
                
                fname_input_list.append(fn)
        
    else:
        fname_input_list = get_list_of_files(path_input)
    path_fname_csv = os.path.join(path, fname_csv)
    if not os.path.exists(path_fname_csv):
        path_fname_csv = os.path.join(path, 'tf.csv')

    R_board2world, t_board2world, R_cam2board, t_cam2board = get_board2world_cam2board_from_csv(fname_csv=path_fname_csv, camera_name=camera_name)
    n_Rt_matrices = 0
    for fname_input in fname_input_list:
        fname_output = fname_input
#        fname_output = str(n).zfill(6) + '.txt'

        t_controller2world, R_controller2world = extract_tra_rotquat(os.path.join(path_input, fname_input))
        Rt = Rt_end2cam(R_controller2world, t_controller2world, R_board2world, t_board2world, R_cam2board, t_cam2board)

        np.savetxt(os.path.join(path_output, fname_output), Rt, fmt='%.7f')
        n_Rt_matrices += 1

    print('=== Rt matrices generated ===\n')
    return n_Rt_matrices, [path_fname_csv,fname_input_list]


def check_timestamps(path, time_difference_threshold=0.010, folder_rgb='Images', folder_pose='6DOF'):
    """Select files with difference between image and pose timestamps below a threshold
    :param path: input folder
    :param time_difference_threshold:
    :param folder_rgb: subfolder with RGB images
    :param folder_pose: subfolder with pose files
    :return: number of selected files
    """
    
    #    CHANGE: make two file lists, one for camera 1 (C1), one for camera 2 (C2)
    
#    import os

    print('\n=== Checking timestamps ===')

    path_rgb = os.path.join(path, folder_rgb)
    path_pose = os.path.join(path, folder_pose)

    fname_selected_files_C1 = os.path.join(path, 'selected_files_C1.txt')
    fname_selected_files_C2 = os.path.join(path, 'selected_files_C2.txt')
    
    fname_rgb_list = get_list_of_files(path_rgb, sort_alphanum=True)
    fname_pose_list = get_list_of_files(path_pose, sort_alphanum=True)

    time_rgb_list = []
    time_pose_list = []
    for i in range(len(fname_pose_list)):
        fname_pose = os.path.join(path_pose, fname_pose_list[i])
        time_rgb, time_pose = extract_timestamps(fname_pose)
        time_rgb_list.append(time_rgb)
        time_pose_list.append(time_pose)

    n_selected_files = 0
    
    
    selected_files_list_C1 = []
    selected_files_list_C2 = []
    for i in range(len(time_pose_list)):
        time_difference = abs(time_rgb_list[i] - time_pose_list[i])
        if time_difference > time_difference_threshold:  # difference above threshold
            print('Failed timestamp check:', fname_rgb_list[i])
            continue
        # if (np.argmin(abs(np.asarray(time_rgb_list) - time_pose_list[i]))) != i:  # not the closest timestamp
        #     print 'Failed timestamp check #2:', fname_rgb_list[i]
        #     continue
        if 'C1' in fname_pose_list[i][:-4]:
            
            selected_files_list_C1.append(fname_pose_list[i][:-4])
            
        else:
            
            selected_files_list_C2.append(fname_pose_list[i][:-4])
            
        n_selected_files += 1
        
    with open(fname_selected_files_C1, 'w') as f:
        f.writelines("\n".join(selected_files_list_C1))
        
    with open(fname_selected_files_C2, 'w') as f:
        f.writelines("\n".join(selected_files_list_C2))
        
        
#    print(str(len(selected_files_list)), '/', str(len(fname_pose_list)), 'files passed timestamp consistency check')

#    print('=== Timestamps checked ===\n')

#    return n_selected_files


def get_object_bbox3d(object_name='gluegun'):
#    import numpy as np
    bb_3d = np.zeros((8, 3), float)
    bbox_x = None
    bbox_y = None
    bbox_z = None
    if object_name == 'gluegun':
        # Bounding box v souradne soustave hrotu pistole:
        # osa x ... dopredu ve smeru hrotu
        # osa y ... dolu
        # osa z ... doleva
        gluegun_length = 0.265
        # gluegun_height = 0.315
        gluegun_width = 0.12
        from_endpoint_to_top = 0.135
        from_endpoint_to_bottom = 0.18
        bbox_x = [0., -gluegun_length]
        bbox_y = [from_endpoint_to_bottom, -from_endpoint_to_top]
        bbox_z = [gluegun_width / 2., -gluegun_width / 2.]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if j == 0:
                    bb_3d[4*i + 2*j + k] = [bbox_x[1-i], bbox_y[1-j], bbox_z[1-k]]
                else:
                    bb_3d[4*i + 2*j + k] = [bbox_x[1-i], bbox_y[1-j], bbox_z[k]]
                    
    #    point A (top left point frontal point of 3d bbox)
    
    point_A = bb_3d[6:7,:]
    
#    centroid
    
    centroid = np.zeros((1,3),float)
    
    centroid[:,0] = point_A[:,0] - (gluegun_length/2)
    centroid[:,1] = point_A[:,1] - from_endpoint_to_bottom
    centroid[:,2] = point_A[:,2] - (gluegun_width/2)
                    
    return bb_3d,centroid


def get_internal_calibration_matrix(object_name='gluegun'):
#    import numpy as np
    internal_calibration_matrix = None
    if object_name == 'gluegun':
        internal_calibration_matrix = np.asarray([[543.5049721588473, 0.0, 324.231500840495], [0.0, 541.6044368823277, 244.3817928406171], [0.0, 0.0, 1.0]], dtype='float32')
    return internal_calibration_matrix


def project_from_3d_to_2d(points_3d, transformation, internal_calibration_matrix):
    """
    From BB8
    :param points_3d:
    :param transformation:
    :param internal_calibration_matrix:
    :return:
    """
#    import numpy as np
#    ak chcene ziskat koordinaty 3d bboxu tak vynasobime transoframtion a points_3d
    projections_2d = np.zeros((2, points_3d.shape[1]), dtype='float32')
    camera_projection = (internal_calibration_matrix.dot(transformation)).dot(points_3d)
    bb3d_camera = transformation.dot(points_3d)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d,bb3d_camera

def return_bb_all_objects(path_pose,image_folder):
    
    translation_rot_mat_files = os.listdir(path_pose)
    
    internal_calibration_matrix = get_internal_calibration_matrix()
        
    bb_3d = get_object_bbox3d()
    bb_3d_i = np.c_[bb_3d, np.ones((bb_3d.shape[0], 1))].transpose()
    
#    3d_bboxes = []
    bboxes_2d = []
    image_files = []
    
    for f in translation_rot_mat_files:
        
        if f.endswith('.txt'):
            
            Rt_gt = np.loadtxt(os.path.join(path_pose, f), dtype='float32')[:3, :]
    
            bb_gt = project_from_3d_to_2d(bb_3d_i, Rt_gt, internal_calibration_matrix)
    
            image_name = f.split('.')[0] + '.png'
            
            image_file = os.path.join(image_folder,image_name)
            
            bb_gt = np.rollaxis(bb_gt,axis = 1, start = 0)
            
            bboxes_2d.append(bb_gt)
            image_files.append(image_file)
            
    return bboxes_2d,image_files

