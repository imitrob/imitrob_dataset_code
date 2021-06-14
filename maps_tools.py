#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:34:47 2019

@author: matusmacbookpro
"""
import numpy as np
from PIL import Image
from PIL import ImageDraw
from math import acos
from math import sqrt
from math import pi
import colorsys
import copy

#adapted from: https://github.com/NVlabs/Deep_Object_Pose/blob/master/scripts/train.py

def CreateBeliefMap(img_size,points_belief,nbpoints = 9,sigma=2,scale = 8):
    """
    Args: 
        img_size: list of image sizes
        pointsBelief: list of points in the form of 
                      [nb object, nb points, 2 (x,y)], nested array, nb points is allways 8, nb object - number of objects in screen regardless of category
        nbpoints: (int) number of points, DOPE uses 8 points here
        sigma: (int) size of the belief map point
    return: 
        return an array of PIL black and white images representing the 
        belief maps         
    """
    pointsBelief = copy.deepcopy(points_belief)
    
    
#    scale points belief for appropriate output size
    for i in range(len(pointsBelief[0])):
    
        pointsBelief[0][i] = (pointsBelief[0][i][0]/scale,pointsBelief[0][i][1]/scale)
        
    img_size[0] = int(img_size[0]/scale)
    img_size[1] = int(img_size[1]/scale)
    
    
    beliefsImg = []
    sigma = int(sigma)
    for numb_point in range(nbpoints):    
        array = np.zeros(img_size)
        out = np.zeros(img_size)

        for point in pointsBelief:
            p = point[numb_point]
            w = int(sigma*2)
            if p[0]-w>=0 and p[0]+w<img_size[0] and p[1]-w>=0 and p[1]+w<img_size[1]:
                for i in range(int(p[0])-w, int(p[0])+w):
                    for j in range(int(p[1])-w, int(p[1])+w):
                        array[i,j] = np.exp(-(((i - p[0])**2 + (j - p[1])**2)/(2*(sigma**2))))

        stack = np.stack([array,array,array],axis=0).transpose(2,1,0)
#        imgBelief = Image.new(img_mode, img_size, "black")
        beliefsImg.append(Image.fromarray((stack*255).astype('uint8')))
        
    bi_concat = []

    for bi in beliefsImg:
        
        bi_concat.append(np.array(bi)[:,:,0:1])
        
    belief_concat = np.concatenate(tuple(bi_concat),axis = 2)
    
    belief_concat = np.rollaxis(belief_concat,2,0)
        
    return belief_concat


def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def length(v):
    return sqrt(v[0]**2+v[1]**2)

def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]

def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]

def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees

def py_ang(A, B=(1,0)):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner


def getAfinityCenter(width, height, point, center, radius=7, img_affinity=None):
    """
    Function to create the affinity maps, 
    e.g., vector maps pointing toward the object center. 
    Args:
        width: image wight
        height: image height
        point: (x,y) 
        center: (x,y)
        radius: pixel radius
        img_affinity: tensor to add to 
    return: 
        return a tensor
    """
    tensor = np.zeros((2,height,width))

    # Create the canvas for the afinity output
    imgAffinity = Image.new("RGB", (width,height), "black")
#    totensor = transforms.Compose([transforms.ToTensor()])
    
    draw = ImageDraw.Draw(imgAffinity)    
    r1 = radius
    p = point
    draw.ellipse((p[0]-r1,p[1]-r1,p[0]+r1,p[1]+r1),(255,255,255))

    del draw

    # Compute the array to add the afinity
    array = (np.array(imgAffinity)/255)[:,:,0]

    angle_vector = np.array(center) - np.array(point)
    angle_vector = normalize(angle_vector)
    affinity = np.concatenate([[array*angle_vector[0]],[array*angle_vector[1]]])

    # print (tensor)
    if not img_affinity is None:
        # Find the angle vector
        # print (angle_vector)
        if length(angle_vector) >0:
            angle=py_ang(angle_vector)
        else:
            angle = 0
        # print(angle)
        c = np.array(colorsys.hsv_to_rgb(angle/360,1,1)) * 255
        draw = ImageDraw.Draw(img_affinity)    
        draw.ellipse((p[0]-r1,p[1]-r1,p[0]+r1,p[1]+r1),fill=(int(c[0]),int(c[1]),int(c[2])))
        del draw
#    re = torch.from_numpy(affinity).float() + tensor
        
    re = affinity + tensor
        
    return re, img_affinity

def GenerateMapAffinity(img_size,img_mode,nb_vertex,points_belief,centroids2d,scale = 8,radius = 1):
    """
    Function to create the affinity maps, 
    e.g., vector maps pointing toward the object center. 
    Args:
        img_size: PIL image
        nb_vertex: (int) number of points 
        pointsInterest: list of points 
        objects_centroid: (x,y) centroids for the obects
        scale: (float) by how much you need to scale down the image 
    return: 
        return a list of tensors for each point except centroid point      
    """

    pointsInterest = copy.deepcopy(points_belief)
    objects_centroid = copy.deepcopy(centroids2d)

#scale points_belief
    
    for i in range(len(pointsInterest[0])):
    
        pointsInterest[0][i] = (pointsInterest[0][i][0]/scale,pointsInterest[0][i][1]/scale)

#scale centroids
        
    objects_centroid = objects_centroid/scale
    
    objects_centroid = [tuple(objects_centroid[0,:])]
    

    # Apply the downscale right now, so the vectors are correct. 
    img_affinity = Image.new(img_mode, (int(img_size[0]/scale),int(img_size[1]/scale)), "black")
    # Create the empty tensors
#    totensor = transforms.Compose([transforms.ToTensor()])

    affinities = []
    for i_points in range(nb_vertex):
#        affinities.append(torch.zeros(2,int(img.size[1]/scale),int(img.size[0]/scale)))
        affinities.append(np.zeros((2,int(img_size[1]/scale),int(img_size[0]/scale))))

    
    for i_pointsImage in range(len(pointsInterest)):    
        pointsImage = pointsInterest[i_pointsImage]
        center = objects_centroid[i_pointsImage]
        for i_points in range(nb_vertex):
            point = pointsImage[i_points]
            affinity_pair, img_affinity = getAfinityCenter(int(img_size[0]/scale),
                int(img_size[1]/scale),
                tuple((np.array(pointsImage[i_points])/1).tolist()),
                tuple((np.array(center)/1).tolist()), 
                img_affinity = img_affinity, radius=radius)

            affinities[i_points] = (affinities[i_points] + affinity_pair)/2


            # Normalizing
            v = affinities[i_points]                    
            
            xvec = v[0]
            yvec = v[1]

            norms = np.sqrt(xvec * xvec + yvec * yvec)
            nonzero = norms > 0

            xvec[nonzero]/=norms[nonzero]
            yvec[nonzero]/=norms[nonzero]
            
            affinities[i_points] = np.concatenate([[xvec],[yvec]])

    affinities = np.concatenate(affinities,0)

    return affinities