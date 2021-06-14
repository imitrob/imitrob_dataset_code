#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:35:57 2019

@author: matusmacbookpro
"""

#adapted from: https://github.com/NVlabs/Deep_Object_Pose/blob/master/src/dope/inference/cuboid_pnp_solver.py

import cv2
import numpy as np
from pyrr import Quaternion


class CuboidPNPSolver():
    """
    This class is used to find the 6-DoF pose of a cuboid given its projected vertices.
    Runs perspective-n-point (PNP) algorithm.
    """

#   INIT PARAMETERS:
    
#   camera_intrinsic_matrix: size (3,3)
#   cuboid3d : default dimensions of object cuboid in 3d (canonical locations of every vertex + centroid), size (9,3)
#    default cuboid3d for glue gun:
#    array([[-0.265 , -0.135 , -0.06  ],
#       [-0.265 , -0.135 ,  0.06  ],
#       [-0.265 ,  0.18  ,  0.06  ],
#       [-0.265 ,  0.18  , -0.06  ],
#       [ 0.    , -0.135 , -0.06  ],
#       [ 0.    , -0.135 ,  0.06  ],
#       [ 0.    ,  0.18  ,  0.06  ],
#       [ 0.    ,  0.18  , -0.06  ],
#       [-0.1325,  0.    ,  0.    ]])
    
#    dist_coeffs: camera distortion coefficients, size: (4,1), by default set to zeros, should work fine if you dont use camera with ridiculous distortion (i.e. gopro)
    


    # Class variables
    cv2version = cv2.__version__.split('.')
    cv2majorversion = int(cv2version[0])

    def __init__(self, camera_intrinsic_matrix = None, cuboid3d = None, dist_coeffs = np.zeros((4, 1))):
        
        if (not camera_intrinsic_matrix is None):
            self._camera_intrinsic_matrix = camera_intrinsic_matrix
        else:
            self._camera_intrinsic_matrix = np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ])
        self._cuboid3d = cuboid3d
        
        self._dist_coeffs = dist_coeffs

    def set_camera_intrinsic_matrix(self, new_intrinsic_matrix):
        '''Sets the camera intrinsic matrix'''
        self._camera_intrinsic_matrix = new_intrinsic_matrix

    def set_dist_coeffs(self, dist_coeffs):
        '''Sets the camera intrinsic matrix'''
        self._dist_coeffs = dist_coeffs

    def solve_pnp(self, cuboid2d_points, pnp_algorithm = None):
        """
        Detects the rotation and traslation 
        of a cuboid object from its vertexes' 
        2D location in the image
        """
        
#       INPUT:
#       cuboid2d_points: size (9,2)
#       cuboid2d contains the 2d coordinates of bb vertcies + the 2d coordinates of the bb centroid
#       last coordinate is the centroid
#       pnp_algorithm: type of pnp algotithm used, if None uses cv2.SOLVEPNP_ITERATIVE
        
#        OUTPUT: 
#        location: [x,y,z]
#        quaternion: [x,y,z,w]
#        projected_points (9x2) (DONT USE THIS, sometimes gives incorrect output)
#        RTT_matrix (3x4)
        

        # Fallback to default PNP algorithm base on OpenCV version
        if pnp_algorithm is None:
            if CuboidPNPSolver.cv2majorversion == 2:
                pnp_algorithm = cv2.CV_ITERATIVE
            elif CuboidPNPSolver.cv2majorversion == 3:
                pnp_algorithm = cv2.SOLVEPNP_ITERATIVE
#                pnp_algorithm = cv2.SOLVE_PNP_P3P
            else:
#                pnp_algorithm = cv2.SOLVEPNP_ITERATIVE
                # Alternative algorithms:
                # pnp_algorithm = SOLVE_PNP_P3P  
                pnp_algorithm = cv2.SOLVEPNP_EPNP        
        
        location = None
        quaternion = None
        projected_points = cuboid2d_points

        cuboid3d_points = np.array(self._cuboid3d)
        obj_2d_points = []
        obj_3d_points = []

        for i in range(9):
            check_point_2d = cuboid2d_points[i]
            # Ignore invalid points
            if (check_point_2d is None):
                continue
            obj_2d_points.append(check_point_2d)
            obj_3d_points.append(cuboid3d_points[i])

        obj_2d_points = np.array(obj_2d_points, dtype=np.float64)
        obj_3d_points = np.array(obj_3d_points, dtype=np.float64)

        valid_point_count = len(obj_2d_points)

        # Can only do PNP if we have more than 3 valid points
        is_points_valid = valid_point_count >= 4

        if is_points_valid:
            
            ret, rvec, tvec = cv2.solvePnP(
                obj_3d_points,
                obj_2d_points,
                self._camera_intrinsic_matrix,
                self._dist_coeffs,
                flags=pnp_algorithm
            )
            
#            ret, rvec, tvec,_ = cv2.solvePnPRansac(
#                obj_3d_points,
#                obj_2d_points,
#                self._camera_intrinsic_matrix,
#                self._dist_coeffs,
#                reprojectionError=16
#            )

            if ret:
                location = list(x[0] for x in tvec)
                quaternion = self.convert_rvec_to_quaternion(rvec)
                
                projected_points, _ = cv2.projectPoints(cuboid3d_points, rvec, tvec, self._camera_intrinsic_matrix, self._dist_coeffs)
                projected_points = np.squeeze(projected_points)
                
                # If the location.Z is negative or object is behind the camera then flip both location and rotation
                x, y, z = location
                if z < 0:
                    # Get the opposite location
                    location = [-x, -y, -z]

                    # Change the rotation by 180 degree
                    rotate_angle = np.pi
                    rotate_quaternion = Quaternion.from_axis_rotation(location, rotate_angle)
                    quaternion = rotate_quaternion.cross(quaternion)
                    
#        calculate rotation and translation matrix from quaternion and location
                    
        rt_matrix = quaternion.matrix33
        RTT_matrix = np.concatenate((rt_matrix,np.array(location).reshape(3,1)),axis=1)

        return location, quaternion, projected_points,RTT_matrix

    def convert_rvec_to_quaternion(self, rvec):
        '''Convert rvec (which is log quaternion) to quaternion'''
        theta = np.sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2])  # in radians
        raxis = [rvec[0] / theta, rvec[1] / theta, rvec[2] / theta]

        # pyrr's Quaternion (order is XYZW), https://pyrr.readthedocs.io/en/latest/oo_api_quaternion.html
        return Quaternion.from_axis_rotation(raxis, theta)
        
        # Alternatively: pyquaternion
        # return Quaternion(axis=raxis, radians=theta)  # uses OpenCV's Quaternion (order is WXYZ)

    def project_points(self, rvec, tvec):
        '''Project points from model onto image using rotation, translation'''
        output_points, tmp = cv2.projectPoints(
            self.__object_vertex_coordinates, 
            rvec, 
            tvec, 
            self.__camera_intrinsic_matrix, 
            self.__dist_coeffs)
        
        output_points = np.squeeze(output_points)
        return output_points