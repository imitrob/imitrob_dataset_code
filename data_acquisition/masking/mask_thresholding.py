import cv2
import os
import numpy as np
import json
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage.filters import gaussian_filter
import argparse
import shutil


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

def crop_by_bbox(mask, folder_bbox, frame, camera):

    fname_bbox = os.path.join(folder_bbox, '{}.json'.format(frame))
    bbox_vertices = bbox2d_from_json(fname_bbox, frame, camera)

    bbox_hull = ConvexHull(bbox_vertices)
    bbox_hull_points = bbox_hull.points[bbox_hull.vertices, :]
    bbox_dhp = Delaunay(bbox_hull_points)

    h = np.shape(mask)[0]
    w = np.shape(mask)[1]
    points_test = np.transpose(np.meshgrid(range(h), range(w)))
    points_test = np.reshape(points_test, [h * w, 2])
    points_inside_hull = np.asarray(bbox_dhp.find_simplex(points_test) >= 0)
    mask = mask * np.reshape(points_inside_hull, [h, w])
    return mask

def mask_out_hand(rgb, tolerance=10):
    mask_wo_hand = (rgb[:, :, 0] < np.max([rgb[:, :, 1], rgb[:, :, 2]], axis=0) + tolerance)
    return mask_wo_hand

def in_poly_hull_multi(poly, points, hull=None):
    if hull is None:
        hull = ConvexHull(poly)
    res = []
    for p in points:
        new_hull = ConvexHull(np.concatenate((poly, [p])))
        res.append(np.array_equal(new_hull.vertices, hull.vertices))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("tool-name", choices=['gluegun', 'hammer', 'gluegun', 'roller'], help="Name of the tool to be processed.")
    parser.add_argument("data-folder", type=str, help="Path to the folder containing the extracted recordings.")
    parser.add_argument("--recompute-bg-values", action="store_true", default=False, help="Instead of loading BG RGB data from file, recompute them.")
    parser.add_argument('--remove-hand', action='store_true', default=False, help='Remove also the hand from the mask.')
    args = parser.parse_args()

    tool_name = args.tool_name
    path_bg = 'compute_bg_{}'.format(tool_name)  # Folder for computation of background.
    bg_rgb_values_from_file = not args.recompute_bg_values  # False == recompute.
    remove_hand = args.remove_hand  # False == keep hand. True == mask out hand.

    path = args.data_folder
    folder_list = get_list_of_folders(path, sort=True)
    subfolder_img = 'Image'
    subfolder_bbox = 'BBox'
    if remove_hand:
        subfolder_mask = 'Mask_thresholding_wo_hand'
    else:
        subfolder_mask = 'Mask_thresholding'

    for folder in folder_list:
        if tool_name not in folder:
            continue
        if 'C1' in folder:
            camera = 'C1'
        elif 'C2' in folder:
            camera = 'C2'
        else:
            continue

        fname_bg_rgb_values = os.path.join(path_bg, 'bg_rgb_values_{}.npy'.format(camera))
        if bg_rgb_values_from_file and os.path.isfile(fname_bg_rgb_values):
            # Load background pixel values
            bg_rgb_values = np.load(fname_bg_rgb_values)
        else:
            # Load bg mask (i.e. green)
            fname_mask_bg = os.path.join(path_bg, '{}_mask_bg_safe.png'.format(camera))
            if not os.path.isfile(fname_mask_bg):
                fname_mask_bg = os.path.join(path_bg, '{}_mask_bg.png'.format(camera))
            mask_bg = cv2.imread(fname_mask_bg, 0)
            # Load images for bg mask
            bg_rgb_values = None
            for i in range(100):
                fname_img = os.path.join(path_bg, subfolder_img, '{}F{:04d}.png'.format(camera, i))
                if not os.path.isfile(fname_img):
                    continue
                rgb = cv2.imread(fname_img, 1)[:, :, ::-1]
                bg_rgb_values_i = rgb[mask_bg >= 128]
                if bg_rgb_values is None:
                    bg_rgb_values = bg_rgb_values_i
                else:
                    bg_rgb_values = np.vstack([bg_rgb_values, bg_rgb_values_i])
            # Save background pixel values
            np.save(fname_bg_rgb_values, bg_rgb_values)

        # Convex hull for bg values
        hull = ConvexHull(bg_rgb_values)
        hull_points = hull.points[hull.vertices, :]
        #print(hull_points)
        #print(np.shape(hull_points))  # number of convex hull vertices

        # Add tolerance to bg values
        factor = 10  # TODO
        bg_mean = np.mean(bg_rgb_values, 0)
        d = hull_points - bg_mean
        d = (d.T / np.linalg.norm(d, axis=1)).T
        hull_points_1 = hull_points + factor * d

        # Convex hull for bg values
        hull_1 = ConvexHull(hull_points_1)
        hull_points_1 = hull_1.points[hull_1.vertices, :]
        dhp_1 = Delaunay(hull_points_1)

        # Load outside mask (i.e. neither bg nor object/hand)
        fname_mask_valid = os.path.join(path_bg, '{}_mask_bg.png'.format(camera))
        mask_valid = cv2.imread(fname_mask_valid, 0)

    #    os.makedirs(os.path.join(path, folder, subfolder_mask), exist_ok=True)
        path_output = os.path.join(path, folder, subfolder_mask)
        if os.path.exists(path_output):
            shutil.rmtree(path_output)
        os.makedirs(path_output)

        print('Camera: {}\tFolder: {}'.format(camera, folder))

        frame_list = [os.path.splitext(basename)[0][2:] for basename in get_list_of_files(os.path.join(path, folder, subfolder_img), sort=True) if basename[:2] == camera]

        for frame in frame_list:

            fname_img = os.path.join(path, folder, subfolder_img, '{}{}.png'.format(camera, frame))
            fname_bbox = os.path.join(path, folder, subfolder_bbox, '{}.json'.format(frame))
            if not os.path.isfile(fname_img) or not os.path.isfile(fname_bbox):
                continue

            rgb = cv2.imread(fname_img, 1)[:, :, ::-1]
    #        print(frame)

            points_test = np.reshape(rgb, [np.shape(rgb)[0] * np.shape(rgb)[1], np.shape(rgb)[2]])
            points_outside_hull = np.asarray(dhp_1.find_simplex(points_test) < 0)  # fg

            mask = np.reshape(points_outside_hull, [np.shape(rgb)[0], np.shape(rgb)[1]])  # fg mask

            if remove_hand:
                mask_wo_hand = mask_out_hand(rgb)
                mask = mask * mask_wo_hand

            mask[mask_valid < 128] = 0  # remove area outside green cloth
            mask = crop_by_bbox(mask, os.path.join(path, folder, subfolder_bbox), frame, camera)  # remove everything outside bbox

            mask = np.asarray(255 * mask, np.uint8)

            cv2.imwrite(os.path.join(path, folder, subfolder_mask, '{}{}.png'.format(camera, frame)), mask)

