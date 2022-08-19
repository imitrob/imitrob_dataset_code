import os
import shutil
import numpy as np
import cv2
import json
from scipy.spatial import ConvexHull, Delaunay

def alphanum_key(s):
    import re
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    return [tryint(c) for c in re.split('([0-9]+)', s)]

def get_list_of_files(path, sort=True):
    basename_list = [basename for basename in os.listdir(path) if os.path.isfile(os.path.join(path, basename))]
    if sort:
        basename_list.sort(key=alphanum_key)
    return basename_list

def get_list_of_folders(path, sort=True):
    basename_list = [basename for basename in os.listdir(path) if os.path.isdir(os.path.join(path, basename))]
    if sort:
        basename_list.sort(key=alphanum_key)
    return basename_list

def bbox2d_from_json(fname_bbox, frame, camera):
    with open(fname_bbox) as f:
        data = json.load(f)
    bbox_json = data[frame]['BBox_2D_{}'.format(camera)]
    bbox_array = np.zeros((8, 2))
    for i in range(8):
        bbox_array[i, 0] = bbox_json['V{}'.format(i+1)]['v']
        bbox_array[i, 1] = bbox_json['V{}'.format(i+1)]['u']
    return bbox_array

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

def visualize_bbox(path, subfolder_img='Image', subfolder_bbox='BBox', subfolder_output='BBox_visualized', bbox_radius=6):
    # colors of bbox vertices:
    colors = [[  0,255,  0,255,  0,255,  0,255,127],
              [  0,  0,255,255,  0,  0,255,255,127],
              [  0,  0,  0,  0,255,255,255,255,127]]
    borders = [0, 0, 0, 0, 0, 0, 0, 0]
    r = bbox_radius

    path_img = os.path.join(path, subfolder_img)
    path_bbox = os.path.join(path, subfolder_bbox)
    path_output = os.path.join(path, subfolder_output)
#    os.makedirs(path_output, exist_ok=True)
    if os.path.exists(path_output):
        shutil.rmtree(path_output)
    os.makedirs(path_output)
    
    basename_img_list = get_list_of_files(path_img, sort=True)
    for basename_img in basename_img_list:
        fname_img = os.path.join(path_img, basename_img)  # fname_img ... RGB file name including folders
        camera = basename_img[:2]  # e.g. 'C1'
        frame = basename_img[2:-4]  # e.g. 'F0000'
        fname_bbox = os.path.join(path_bbox, '{}.json'.format(frame))
        if not os.path.isfile(fname_bbox):
            continue
        bbox_array = bbox2d_from_json(fname_bbox, frame, camera)  # extract 2D bbox from json as numpy array (8, 2)
        bbox_array = np.asarray(np.round(bbox_array), int)
        img = cv2.imread(fname_img, 1)[:, :, ::-1]
        img = highlight_bbox(img, bbox_array)
        h, w = np.shape(img)[:2]
        for i in range(8):
            y, x = bbox_array[i, :2]
            if -r <= x < w + r and -r <= y < h + r:
                for k in range(3):
                    img[max(y-r, 0):min(y+r+1, h), max(x-r, 0):min(x+r+1, w), k] = borders[i]
                    img[max(y-r+1, 0):min(y+r, h), max(x-r+1, 0):min(x+r, w), k] = colors[k][i]
        fname_output = os.path.join(path_output, basename_img)
        cv2.imwrite(fname_output, img[:, :, ::-1])
    return

bbox_radius = 6  # TODO

path = os.path.join('recordings', 'dataset_links')
folder_list = get_list_of_folders(path, sort=True)
subfolder_img = 'Image'
subfolder_bbox = 'BBox'
subfolder_output = 'BBox_visualization'

rewrite_existing_subfolder_output = True  # TODO
tool_name = 'gluegun'  # TODO

for folder in folder_list:

    if tool_name not in folder:  # TODO
        continue

    if os.path.isdir(os.path.join(path, folder, subfolder_output)) and not rewrite_existing_subfolder_output:
        print('Warning: Skipping the following folder because it already contains {}: {}'.format(subfolder_output, os.path.join(path, folder)))
        continue
    if not os.access(os.path.join(path, folder), os.W_OK):
        print('ERROR: Skipping the following folder because it is not writable: {}'.format(os.path.join(path, folder)))
        continue

    print('Processing folder: {}'.format(os.path.join(path, folder)))
    visualize_bbox(os.path.join(path, folder), subfolder_img, subfolder_bbox, subfolder_output, bbox_radius)

