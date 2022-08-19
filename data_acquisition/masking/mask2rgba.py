import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import sys
import shutil


###
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

def create_circular_mask(h, w=None, radius=None):
    if w is None:
        w = h
    center = (w//2, h//2)
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1]) + 0.5
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center < radius
    mask = np.asarray(mask, np.uint8)
    return mask

def read_trimap_with_border(name, border_width=15):
    trimap_im = cv2.imread(name, 0) / 255
    h, w = trimap_im.shape
    kernel_ci = create_circular_mask(border_width)
    mask_outer = np.copy(trimap_im)
    mask_inner = np.copy(trimap_im)
    mask_outer = cv2.dilate(mask_outer, kernel_ci, iterations=1)
    mask_inner = cv2.erode(mask_inner, kernel_ci, iterations=1)
    trimap = np.zeros((h, w, 2))
    trimap[mask_inner == 1, 1] = 1
    trimap[mask_outer == 0, 0] = 1
    return trimap


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates masks of a tool from extracted recordings.')
    parser.add_argument("tool-name", choices=['gluegun', 'hammer', 'gluegun', 'roller'], help="Name of the tool to be processed.")
    parser.add_argument("data-folder", type=str, help="Path to the folder containing the extracted recordings.")
    parser.add_argument('--overwrite-existing', action='store_true', default=False, help='Overwrite existing masking outputs.')
    parser.add_argument('--remove-hand', action='store_true', default=False, help='Remove also the hand from the mask.')
    parser.add_argument('--border-width', type=int, default=11)
    parser.add_argument('--fba-path', type=str, default=None, help='Path to the cloned FBA Matting repository. Do not specify if this script is being run from within the repo folder.')
    parser.add_argument('--weights-absolute', action='store_true', default=False, help="Use this flag to specify absolute path to the FBA weights.")
    parser.add_argument('--fba-weights', type=str, default='FBA.pth', help='Path to the FBA weights. By default relative to the FBA repository.')
    args = parser.parse_args()

    class FBAMattingArgs:
        encoder = 'resnet50_GN_WS'
        decoder = 'fba_decoder'
        if args.weights_relative:
            weights = args.fba_weights
        else:
            weights = os.path.join(args.fba_path, args.fba_weights)

    fba_matting_args=FBAMattingArgs()

    sys.path.insert(1, args.fba_path)
    try:
        from demo import np_to_torch, pred, scale_input
        from dataloader import read_image, read_trimap
        from networks.models import build_model
    except ImportError:
        print(f"Could not import FBA Matting files, please make sure the path is correct.\nSpecified import path: {args.fba_path}")
        print(f"If you don't have FBA Matting repo cloned, please, clone it from here: https://github.com/MarcoForte/FBA_Matting")
        sys.exit(-1)
    model = build_model(fba_matting_args)

    tool_name = args.tool_name

    border_width = args.border_width

    path = args.data_folder
    folder_list = get_list_of_folders(path, sort=True)
    subfolder_rgb = 'Image'

    remove_hand = args.remove_hand  # False == keep hand. True == mask out hand.

    if remove_hand:
        subfolder_mask = 'Mask_thresholding_wo_hand'
        subfolder_output = 'Mask_wo_hand'
    else:
        subfolder_mask = 'Mask_thresholding'
        subfolder_output = 'Mask'

    rewrite_existing_subfolder_output = args.overwrite_existing

    for folder in folder_list:

        if tool_name not in folder:  # TODO
            continue

        path_rgb = os.path.join(path, folder, subfolder_rgb)
        path_mask = os.path.join(path, folder, subfolder_mask)
        path_output = os.path.join(path, folder, subfolder_output)

        if not os.path.isdir(path_mask):
            print('Warning: The following folder does not contain {}: {}'.format(subfolder_mask, os.path.join(path, folder)))
            continue
        if os.path.isdir(path_output) and not rewrite_existing_subfolder_output:  # TODO
            print('Warning: Skipping the following folder because it already contains {}: {}'.format(subfolder_output, os.path.join(path, folder)))
            continue

        if os.path.exists(path_output):
            shutil.rmtree(path_output)
        os.makedirs(path_output)

        print('Processing folder: {}'.format(os.path.join(path, folder)))

        basename_list = get_list_of_files(path_mask, sort=True)

        for basename in basename_list:
            fname_rgb = os.path.join(path_rgb, basename)
            fname_mask = os.path.join(path_mask, basename)
            fname_output = os.path.join(path_output, basename)

            image = read_image(fname_rgb)
            trimap = read_trimap_with_border(fname_mask, border_width)

            fg, bg, alpha = pred(image, trimap, model)

            output = 255 * np.dstack([fg[:,:,::-1], alpha])

            cv2.imwrite(fname_output, output)
