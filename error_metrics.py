#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:48:57 2019

@author: matusmacbookpro
"""

import numpy as np
import math

#SOURCE: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

#SOURCE: https://www.learnopencv.com/rotation-matrix-to-euler-angles/

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def rot_trans_err(RTT_matrix_gt,RTT_matrix_pred):

    trans_gt = RTT_matrix_gt[:,3]
    rot_gt = RTT_matrix_gt[:,0:3]
    trans_pred = RTT_matrix_pred[:,3]
    rot_pred = RTT_matrix_pred[:,0:3]

#    translation_err = np.linalg.norm(trans_gt-trans_pred)

    squared_dist = np.sum((trans_gt-trans_pred)**2, axis=0)
    translation_err = np.sqrt(squared_dist)

    rot_gt = rotationMatrixToEulerAngles(rot_gt)
    rot_pred = rotationMatrixToEulerAngles(rot_pred)

#    zmenit vypocet rotacie, prehodit rot_gt a rot_pred do eulerovych uhlov, rozdiel medzti tymito dvoma reprezentaciami bude chyba v radianoch
#    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
#    mozno skusit prehodit z radianov do stupnov
    rotation_err = np.linalg.norm(rot_gt-rot_pred)

    rotation_err = math.degrees(rotation_err)

    return translation_err,rotation_err


#see ADD metric in: https://arxiv.org/abs/1711.00199

def ADD_error(bb3d_gt,centroid3d_gt,RTT_matrix_pred,bb3d_defoult,centroid3d_defoult):

    bb3d_defoult_i = np.c_[bb3d_defoult, np.ones((bb3d_defoult.shape[0], 1))].transpose()
    centroid3d_defoult_i = np.c_[centroid3d_defoult, np.ones((centroid3d_defoult.shape[0], 1))].transpose()


    points_3d = np.concatenate((bb3d_defoult_i,centroid3d_defoult_i),axis=1)

    points3d_pred = RTT_matrix_pred.dot(points_3d)

    points3d_gt = np.concatenate((bb3d_gt,centroid3d_gt))


    points3d_pred = np.rollaxis(points3d_pred,1,0)

    squared_dist = np.sum((points3d_gt-points3d_pred)**2, axis=0)
    translation_err = np.sqrt(squared_dist)
    translation_err = np.sum(translation_err)/len(translation_err)


    return translation_err

def calculate_AUC(ADD_errors,dist_threshold):

    acc = np.sum(np.array(ADD_errors, dtype=object) <= dist_threshold) / len(ADD_errors)

    return acc









