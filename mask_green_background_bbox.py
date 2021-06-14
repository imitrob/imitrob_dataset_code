"""
Mask out green background and the outside of gluegun 3D bounding box.

Jiri Sedlar, 2019
Intelligent Machine Perception Project (IMPACT)
http://impact.ciirc.cvut.cz/
CIIRC, Czech Technical University in Prague
"""

import numpy as np
import cv2
import skimage.morphology
import matplotlib.pyplot as plt
from PIL import Image

def get_gluegun_bbox_3d():
    bbox_3d = np.zeros((8,3), float)
    # Bounding box in coordinate system of gluegun endpoint:
    # axis x ... forward in direction of the endpoint
    # axis y ... down
    # axis z ... left
    gluegun_length = 0.265
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
                    bbox_3d[4*i + 2*j + k] = [bbox_x[1-i], bbox_y[1-j], bbox_z[1-k]]
                else:
                    bbox_3d[4*i + 2*j + k] = [bbox_x[1-i], bbox_y[1-j], bbox_z[k]]
    return bbox_3d

def get_gluegun_internal_calibration_matrix():
    internal_calibration_matrix = np.asarray([[543.5049721588473, 0.0, 324.231500840495], [0.0, 541.6044368823277, 244.3817928406171], [0.0, 0.0, 1.0]], dtype='float32')
    return internal_calibration_matrix

def project_points_from_3d_to_2d(points_3d, transformation, internal_calibration_matrix):
    points_2d = np.zeros((2, points_3d.shape[1]), dtype='float32')
    camera_projection = (internal_calibration_matrix.dot(transformation)).dot(points_3d)
    for i in range(2):
        points_2d[i, :] = camera_projection[i, :] / camera_projection[2, :]
    return points_2d

def crop_hand_by_bbox(rgba, pose, bbox_3d=None, internal_calibration_matrix=None):
    if internal_calibration_matrix is None:
        internal_calibration_matrix = get_gluegun_internal_calibration_matrix()

    if bbox_3d is None:
        bbox_3d = get_gluegun_bbox_3d()

    bbox_3d_h = np.c_[bbox_3d, np.ones((bbox_3d.shape[0], 1))].transpose()
    # column ... 3D homogeneous coordinates of 1 vertex

#    rgba = cv2.imread(fname_rgba, -1)
    h = np.shape(rgba)[0]
    w = np.shape(rgba)[1]
    if np.shape(rgba)[2] == 3:
        tmp = 255. * np.ones((h, w, np.shape(rgba)[2] + 1))
        tmp[:, :, :-1] = rgba
        rgba = tmp

#    Rt_gt = np.loadtxt(fname_pose, dtype='float32')[:3, :]
    Rt_gt = pose.astype(np.float32)
    bbox_2d = project_points_from_3d_to_2d(bbox_3d_h, Rt_gt, internal_calibration_matrix)
    
    mask = np.zeros([h, w])
    for j in range(len(bbox_2d[0])):
        x = int(np.round(bbox_2d[0, j]))
        y = int(np.round(bbox_2d[1, j]))
        mask[max(0, min(y, h-1)), max(0, min(x, w-1))] = 1
    mask = skimage.morphology.convex_hull.convex_hull_image(mask)
    mask = np.asarray(mask, float)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    rgba[:, :, 3] = mask * rgba[:, :, 3]
    return rgba, mask

def rgba_from_rgb_green_bg(rgb, r_minus_g_threshold=0, b_minus_g_threshold=-48):
#    rgb = cv2.imread(fname_rgb, 1)
    rgb = np.asarray(rgb, float)
    
    r = rgb[:,:,2]
    g = rgb[:,:,1]
    b = rgb[:,:,0]

    gbr = (2. * g - b - r)
    gbr = (gbr - np.min(gbr)) / (np.max(gbr) - np.min(gbr))

    alpha = gbr

    mask = np.zeros_like(g)  # 1 .. foreground, 0 .. green background
    mask[r - g > r_minus_g_threshold] = 1.
    mask[b - g > b_minus_g_threshold] = 1.

    kernel_ci = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel_sq = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.dilate(mask, kernel_sq, iterations = 1)
    mask = cv2.erode(mask, kernel_ci, iterations = 1)
    mask_inside = cv2.erode(mask, kernel_sq, iterations = 1)
    mask_boundary = mask - mask_inside

    mask = mask_inside + alpha * mask_boundary
    mask = cv2.GaussianBlur(mask, (3,3), 0)

    rgba = np.zeros([np.shape(rgb)[0], np.shape(rgb)[1], 4])
    rgba[:,:,2] = r
    rgba[:,:,1] = np.min([g, (r+b)/2], axis=0)  # green
    rgba[:,:,0] = b
    rgba[:,:,3] = 255. * mask  # alpha
    return rgba, mask

# DEMO:
#fname = 'C2F0000'
#folder = 'LH_T01/'
#fname_rgb = folder + fname + '.png'  # TODO: Images/
#fname_pose = folder + fname + '.txt'  # TODO: Pose/

##fname_rgb = r'E:\Python 3 projects\tool_recognition_dataset\gun_in_hand_organized\LH\T01\Images\C2F0000.png'
#fname_rgb = r'E:\Python 3 projects\tool_recognition_dataset\LH_T01\C2F0000.png'
#fname_pose = r'E:\Python 3 projects\tool_recognition_dataset\gun_in_hand_organized\LH\T01\Pose\C2F0000.txt'
#
##fname_rgba = folder + fname + '_rgba.png'
##fname_mask = folder + fname + '_mask.png'
#
#rgb_raw = cv2.imread(fname_rgb, 1)
#
#rgba, mask_green = rgba_from_rgb_green_bg(rgb_raw)
#
#plt.imshow(rgba[:, :, :3])
#plt.show()
#
#pose = np.loadtxt(fname_pose, dtype='float32')[:3, :]
#
##cv2.imwrite(fname_rgba, rgba)
#rgba, mask_bbox = crop_hand_by_bbox(rgba, pose)
#
#plt.imshow(rgba[:, :, :3])
#plt.show()
#
##cv2.imwrite(fname_rgba, rgba)
#mask = mask_green * mask_bbox
#
#plt.imshow(mask)
#plt.show()
#
##cv2.imwrite(fname_mask, 255. * mask)
#
#rgb = np.stack([rgba[:, :, 2],rgba[:, :, 1],rgba[:, :, 0]],axis=2).astype(np.uint8)
#
##fname_augmented = folder + fname + '_augm.png'
##color_list = [135, 235, 135]  # random background color
#color_list = [255, 0, 0]  # random background color
#for i in range(len(color_list)):
#    rgb[:, :, i] = mask * rgb[:, :, i] + (1. - mask) * color_list[i]
##cv2.imwrite(fname_augmented, rgba[:, :, :3])
#
##RGB_final = np.stack([rgba[:, :, 2],rgba[:, :, 1],rgba[:, :, 0]],axis=2).astype(np.uint8)
#
#out = Image.fromarray(rgb)
#
#plt.imshow(out)
#plt.show()

#fname_augmented = r'E:\Python 3 projects\tool_recognition_dataset\LH_T01\C2F0000_augumented.png'

#cv2.imwrite(fname_augmented, rgba[:, :, :3])
