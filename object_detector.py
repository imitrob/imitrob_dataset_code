#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:41:42 2019

@author: matusmacbookpro
"""

#adapted from: https://github.com/NVlabs/Deep_Object_Pose/blob/master/src/dope/inference/cuboid_pnp_solver.py

import numpy as np
import copy
from scipy.ndimage.filters import gaussian_filter

"""

The indexes of the 3D bounding cuboid (bb2d) are in the following order:  
- `FrontTopRight` [A], bb2d[0,:]
- `FrontTopLeft` [B], bb2d[1,:]
- `FrontBottomLeft` [C], bb2d[2,:]
- `FrontBottomRight` [D], bb2d[3,:]
- `RearTopRight` [E], bb2d[4,:]
- `RearTopLeft` [F], bb2d[5,:]
- `RearBottomLeft` [G], bb2d[6,:]
- `RearBottomRight` [H], bb2d[7,:]


```
      E +-----------------+ F
       /     TOP         /|
      /                 / |
   A +-----------------+ B|
     |      FRONT      |  |
     |                 |  |
     |  z <--+         |  |
     |       |         |  |
     |       v         |  + G
     |        y        | /
     |                 |/
   D +-----------------+ C
```
"""




def find_objects(vertex2, aff, scale = 1,input_scale = 2, numvertex=8):
    ''' Detects objects given network belief maps and affinities, using heuristic method'''
    
#    INPUT PARAMETERS:
#    vertex2: belief map, size: (9,x,y), where x,y are dimensions of network output,
#    if X,Y are dimensions of network input then x = X/8, y = Y/8
#    aff : affinity map, size: (16,x,y)
#    scale: subsampling parameter, leave at 1, subsampling is handled outside of this code
#    numvertex: number of vertices, leave at 8
    
#    OUTPUTS: 
#    cuboid2d: cuboid2d contains the 2d coordinates of bb vertcies + the 2d coordinates of the bb centroid
#    cuboid2d dimension is (9,2), 8 vertices, last coordinate is the centroid
#    objects: list of potential objects in the net output, each potential object is represented by an list
#    each object list contains 4 entries, first entry are coordinates of a centroid, second entry is a list of vertices
#    at the begining of an training each object can have incomplete information about vertices and centroids which has to be accounted for
#    if this would be used for detecting multiple objects, objects output should contain all found objects, in this applicastion we are using first complete bb found in objects
    
    vertex2 = vertex2.astype(np.float64)
    
    config_sigma = 3
    config_thresh_map = 0.01
    config_thresh_points = 0.1
    config_threshold = 0.5
    config_thresh_angle = 0.5
    

    all_peaks = []
    peak_counter = 0
    for j in range(vertex2.shape[0]):
        belief = copy.deepcopy(vertex2[j])
        map_ori = belief
        
        map = gaussian_filter(belief, sigma=config_sigma)
        p = 1
        map_left = np.zeros(map.shape)
        map_left[p:,:] = map[:-p,:]
        map_right = np.zeros(map.shape)
        map_right[:-p,:] = map[p:,:]
        map_up = np.zeros(map.shape)
        map_up[:,p:] = map[:,:-p]
        map_down = np.zeros(map.shape)
        map_down[:,:-p] = map[:,p:]

        peaks_binary = np.logical_and.reduce(
                            (
                                map >= map_left, 
                                map >= map_right, 
                                map >= map_up, 
                                map >= map_down, 
                                map > config_thresh_map)
                            )
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) 
        
        # Computing the weigthed average for localizing the peaks
        peaks = list(peaks)
        win = 5
        ran = win // 2
        peaks_avg = []
        for p_value in range(len(peaks)):
            p = peaks[p_value]
            weights = np.zeros((win,win))
            i_values = np.zeros((win,win))
            j_values = np.zeros((win,win))
            for i in range(-ran,ran+1):
                for j in range(-ran,ran+1):
                    if p[1]+i < 0 \
                            or p[1]+i >= map_ori.shape[0] \
                            or p[0]+j < 0 \
                            or p[0]+j >= map_ori.shape[1]:
                        continue 

                    i_values[j+ran, i+ran] = p[1] + i
                    j_values[j+ran, i+ran] = p[0] + j

                    weights[j+ran, i+ran] = (map_ori[p[1]+i, p[0]+j])

            # if the weights are all zeros
            # then add the none continuous points
            OFFSET_DUE_TO_UPSAMPLING = 0.4395
#            OFFSET_DUE_TO_UPSAMPLING = 0.
            try:
                peaks_avg.append(
                    (np.average(j_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING, \
                     np.average(i_values, weights=weights) + OFFSET_DUE_TO_UPSAMPLING))
            except:
                peaks_avg.append((p[0] + OFFSET_DUE_TO_UPSAMPLING, p[1] + OFFSET_DUE_TO_UPSAMPLING))
        # Note: Python3 doesn't support len for zip object
        peaks_len = min(len(np.nonzero(peaks_binary)[1]), len(np.nonzero(peaks_binary)[0]))

        peaks_with_score = [peaks_avg[x_] + (map_ori[peaks[x_][1],peaks[x_][0]],) for x_ in range(len(peaks))]

        id = range(peak_counter, peak_counter + peaks_len)

        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += peaks_len

    objects = []

    # Check object centroid and build the objects if the centroid is found
    for nb_object in range(len(all_peaks[-1])):
        if all_peaks[-1][nb_object][2] > config_thresh_points:
            objects.append([
                [all_peaks[-1][nb_object][:2][0],all_peaks[-1][nb_object][:2][1]],
                [None for i in range(numvertex)],
                [None for i in range(numvertex)],
                all_peaks[-1][nb_object][2]
            ])

    # Working with an output that only has belief maps
    if aff is None:
        if len (objects) > 0 and len(all_peaks)>0 and len(all_peaks[0])>0:
            for i_points in range(8):
                if  len(all_peaks[i_points])>0 and all_peaks[i_points][0][2] > config_threshold:
                    objects[0][1][i_points] = (all_peaks[i_points][0][0], all_peaks[i_points][0][1])
    else:
        # For all points found
        for i_lists in range(len(all_peaks[:-1])):
            lists = all_peaks[i_lists]

            for candidate in lists:
                if candidate[2] < config_thresh_points:
                    continue

                i_best = -1
                best_dist = 10000 
                best_angle = 100
                for i_obj in range(len(objects)):
                    center = [objects[i_obj][0][0], objects[i_obj][0][1]]

                    # integer is used to look into the affinity map, 
                    # but the float version is used to run 
                    point_int = [int(candidate[0]), int(candidate[1])]
                    point = [candidate[0], candidate[1]]

                    # look at the distance to the vector field.
                    v_aff = np.array([
                                    aff[i_lists*2, 
                                    point_int[1],
                                    point_int[0]],
                                    aff[i_lists*2+1, 
                                        point_int[1], 
                                        point_int[0]]]) * 10

                    # normalize the vector
                    xvec = v_aff[0]
                    yvec = v_aff[1]

                    norms = np.sqrt(xvec * xvec + yvec * yvec)

                    xvec/=norms
                    yvec/=norms
                        
                    v_aff = np.concatenate([[xvec],[yvec]])

                    v_center = np.array(center) - np.array(point)
                    xvec = v_center[0]
                    yvec = v_center[1]

                    norms = np.sqrt(xvec * xvec + yvec * yvec)
                        
                    xvec /= norms
                    yvec /= norms

                    v_center = np.concatenate([[xvec],[yvec]])
                    
                    # vector affinity
                    dist_angle = np.linalg.norm(v_center - v_aff)

                    # distance between vertexes
                    dist_point = np.linalg.norm(np.array(point) - np.array(center))

                    if dist_angle < config_thresh_angle and (best_dist > 1000 or best_dist > dist_point):
                        i_best = i_obj
                        best_angle = dist_angle
                        best_dist = dist_point

                if i_best is -1:
                    continue
                
                if objects[i_best][1][i_lists] is None \
                        or best_angle < config_thresh_angle \
                        and best_dist < objects[i_best][2][i_lists][1]:
                    objects[i_best][1][i_lists] = ((candidate[0])*scale, (candidate[1])*scale)
                    objects[i_best][2][i_lists] = (best_angle, best_dist)

    if len(objects) > 0:
        
        for obj in objects:
            
            points = obj[1]
            points.append(tuple(obj[0]))
            
            missing_vertices = 0
            
            for point in points:
                
                if point == None:
                    
                    missing_vertices += 1
                    
            if missing_vertices >= 6:
                
                continue
            
            else:
                
                # scale the vertices coorfinates to original size, check whether some vertex is None and skip that vertex
                for p in range(len(points)):
            
                    if points[p] is not None:
                        
                        points[p] = (points[p][0]*8*input_scale,points[p][1]*8*input_scale)
                        
                cuboid2d = np.copy(points)
                
                return cuboid2d,objects
            
        return None,None
    else:
        return None,None
    

