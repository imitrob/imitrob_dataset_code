# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:31:29 2020

@author: Mat Tun
"""
import os
import numpy as np
import glob
import json
import random
import copy
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from dataset_tools import project_from_3d_to_2d
from maps_tools import CreateBeliefMap, GenerateMapAffinity
from randomize_bg import BG_randomizer


class imitrob_dataset(Dataset):

    def __init__(self, data_path, random_bg_path, mode, sample_selection_mode, randomizer_mode, mask_type, crop, flip,
                 test_examples_fraction, attributes_train, attributes_test, randomization_prob=0.5, photo_bg_prob=0.5,
                 scale=2, sigma=2, radius=2, max_height = 480, max_width = 848):
        """Creates an instance of the Imitrob dataset (http://imitrob.ciirc.cvut.cz/imitrobdataset.php).

        Args:
            data_path (str): The path to the dataset.
            random_bg_path (str): The path to a folder containing images to be used for random background augmentation.
            mode (str): Dataset mode - 'test' or 'train' (see PyTorch Dataset documentation).
            sample_selection_mode (str): 'fraction' or 'subset'. If 'fraction' - all types of images are selected for training
                                        and random fraction of are selected as a test set. If 'subset' - specify which types of images are selected for train and test, specify in lists below.
            randomizer_mode (str):  Determines the way background is removed from images: none, bbox, overlay or overlay_noise_bg.
            mask_type (str): The type of mask used during training. 'Mask' or 'Mask_thresholding'.
            crop (bool): Whether random cropping should be used during randomization.
            flip (bool): Whether flipping should be used during randomization
            test_examples_fraction (float): Fraction of the dataset to be used for testing.
            attributes_train (list): A list of parameters to specify which subset of the Imitrob dataset should be used for training.
                                    The following should be set: [dataset_subset, subject, camera, task, hand, object_type, mask_type].
                                    See the dataset website (http://imitrob.ciirc.cvut.cz/imitrobdataset.php#structure) for possible options.
            attributes_test (list): A list of parameters to specify which subset of the Imitrob dataset should be used for testing.
                                    The following should be set: [dataset_subset, subject, camera, task, hand, object_type, mask_type].
                                    See the dataset website (http://imitrob.ciirc.cvut.cz/imitrobdataset.php#structure) for possible options.
            randomization_prob (float, optional): _description_. Defaults to 0.5.
            photo_bg_prob (float, optional): _description_. Defaults to 0.5.
            scale (int, optional): Scaling of the input images as compared to the network input size. Defaults to 2.
            sigma (int, optional): Sigma used to generate a probability distribution around each bbox point positions. Defaults to 2.
            radius (int, optional): Radius used to generate a probability distribution around each bbox point positions. Defaults to 2.
            max_height (int, optional): Height of the mask. Defaults to 480.
            max_width (int, optional): Width of the mask. Defaults to 848.
        """

        # mode determines if we want to return test (mode = 'test'), or train (mode='train')
        self.mode = mode
        # sample_selection_mode = 'fraction'/'subset' if 'fraction' - all types of images are selected for training and
        # random fraction of them are selected as a test set, if 'subset' - specify which types of images
        # are selected for train and test, use attributes_train, attributes_test to do that
        self.sample_selection_mode = sample_selection_mode
        self.mask_type = mask_type
        self.max_height = max_height
        self.max_width = max_width
        # randomizer_mode - determines the way background is removed from image
        # - gbg : remove only green screen
        # - bbox : remove all background
        self.randomizer_mode = randomizer_mode
        self.randomizer_mode_crop = crop
        self.randomizer_mode_flip = flip
        self.randomizer = BG_randomizer(random_bg_path, self.max_width, self.max_height, randomization_prob,
                                        photo_bg_prob)
        self.test_examples_fraction = test_examples_fraction
        self.slash = os.path.sep
        self.source_folder = data_path + self.slash
        # input resolution scaling
        self.scale = scale

        # important parameter, sets the resolution of ground truth, 8 means that the out resolution
        # affinity and belief maps is in_resolution/8
        self.gt_scale = 8 * self.scale
        self.sigma = sigma
        self.radius = radius
        self.dataset_subset = ['Train', 'Test']
        self.subject = ['S1', 'S2', 'S3', 'S4']
        self.camera = ['C1', 'C2']
        self.task = ['random', 'clutter', 'round', 'sweep', 'press', 'frame', 'sparsewave', 'densewave']
        self.hand = ['LH', 'RH']
        self.object_type = ['groutfloat', 'roller', 'gluegun']
        self.paths_images = []
        self.paths_masks = []
        self.paths_t_masks = []
        self.paths_BBox = []
        self.paths_6dof = []
        self.paths_params = []
        self.dataset_subset_list = []
        self.subject_list = []
        self.camera_list = []
        self.task_list = []
        self.hand_list = []
        self.object_type_list = []
        self.im_name_list = []
        self.search_dataset()
        if self.mode == "train":
            self.dataset_subset_train = attributes_train[0]
            self.subject_train = attributes_train[1]
            self.camera_train = attributes_train[2]
            self.task_train = attributes_train[3]
            self.hand_train = attributes_train[4]
            self.object_type_train = attributes_train[5]
            self.mask_type_train = attributes_train[6]
        if self.mode == "test":
            self.dataset_subset_test = attributes_test[0]
            self.subject_test = attributes_test[1]
            self.camera_test = attributes_test[2]
            self.task_test = attributes_test[3]
            self.hand_test = attributes_test[4]
            self.object_type_test = attributes_test[5]
            self.mask_type_test = attributes_test[6]
        self.train_indexes = 0
        self.test_indexes = 0

        if self.sample_selection_mode == 'fraction':
            valid_indexes = self.det_valid_ind(self.subject, self.camera, self.background, self.movement_type,
                                           self.movement_direction, self.object_type)
            self.train_indexes, self.test_indexes = self.split_test_train(valid_indexes, self.test_examples_fraction)
        else:
            if self.mode == "train":
                self.train_indexes = self.det_valid_ind(self.dataset_subset_train, self.subject_train, self.camera_train,
                                               self.task_train, self.hand_train, self.object_type_train)
            if self.mode == "test":
                self.test_indexes = self.det_valid_ind(self.dataset_subset_test, self.subject_test, self.camera_test,
                                                  self.task_test, self.hand_test, self.object_type_test)

    def search_dataset(self):
        for filename in glob.iglob(self.source_folder + '**/*.jpg', recursive=True):
            if 'Image' in filename:
                img_file_list = filename.split(self.slash)
                img_file_name = img_file_list[-1]
                img_file_name = img_file_name.split('.')[0]
                # first two characters in name is camera type
                img_file_cam_type = img_file_name[0:2]
                # rest of the characters are image identifier strating with F
                img_ident = img_file_name[2:]
                # determine path to bbox .json
                bbox_file_name = img_ident + '.json'
                bbox_file_list = copy.deepcopy(img_file_list)
                bbox_file_list[-2] = 'BBox'
                bbox_file_list[-1] = bbox_file_name
                bbox_file_path = os.path.sep.join(bbox_file_list)
                # determine path to 6DOF .json
                sixdof_file_name = img_ident + '.json'
                sixdof_file_list = copy.deepcopy(img_file_list)
                sixdof_file_list[-2] = '6DOF'
                sixdof_file_list[-1] = sixdof_file_name
                sixdof_file_path = os.path.sep.join(sixdof_file_list)

                # determine path to mask .png
                mask_file_name = img_file_name + '.png'
                mask_file_list = copy.deepcopy(img_file_list)
                mask_file_list[-2] = 'Mask'
                mask_file_list[-1] = mask_file_name
                mask_file_path = os.path.sep.join(mask_file_list)

                # determine path to thresholding mask .png
                t_mask_file_name = img_file_name + '.png'
                t_mask_file_list = copy.deepcopy(img_file_list)
                t_mask_file_list[-2] = 'Mask_thresholding'
                t_mask_file_list[-1] = t_mask_file_name
                t_mask_file_path = os.path.sep.join(t_mask_file_list)

                # determine path to parameters file
                parameters_file_name = 'parameters.json'
                parameters_file_list = copy.deepcopy(img_file_list[0:-1])
                parameters_file_list[-1] = parameters_file_name
                parameters_file_path = os.path.sep.join(parameters_file_list)

                # see if bbox file path exists
                try:
                    with open(bbox_file_path) as f:
                        bbox_data = json.load(f)

                    # check if there are any negative values in bbox
                    # if there are skip this file
                    BBox = np.array(
                        [list(val.values()) for val in bbox_data[img_ident]['BBox_2D_' + img_file_cam_type].values()])

                    if np.min(BBox) > 0:
                        self.paths_images.append(filename)
                        self.paths_masks.append(mask_file_path)
                        self.paths_t_masks.append(t_mask_file_path)
                        self.paths_BBox.append(bbox_file_path)
                        self.paths_6dof.append(sixdof_file_path)
                        self.paths_params.append(parameters_file_path)
                        self.im_name_list.append(img_ident)
                        # self.paths_empty_background.append(background_file_path)
                    else:
                        continue
                except:
                    continue

        # make an list of attributes of every data sample
        for i in range(len(self.paths_images)):
            for d in self.dataset_subset:
                if d in self.paths_images[i]:
                    self.dataset_subset_list.append(d)

            for s in self.subject:
                if s in self.paths_images[i]:
                    self.subject_list.append(s)

            for c in self.camera:
                if c in self.paths_images[i]:
                    self.camera_list.append(c)

            # task is not defined for train subset, mark every Train image as having 'random' task
            if self.dataset_subset_list[-1] == 'Train':
                self.task_list.append('random')
            else:
                for t in self.task:
                    if t in self.paths_images[i]:
                        self.task_list.append(t)

            for h in self.hand:
                if h in self.paths_images[i]:
                    self.hand_list.append(h)

            for o in self.object_type:
                if o in self.paths_images[i]:
                    self.object_type_list.append(o)



    # determine valid indexes, ie indexes of the parts that we chose to include
    def det_valid_ind(self, dataset_subset, subject, camera, task, hand, object_type):
        valid_indexes = []
        for i in range(len(self.paths_images)):
            if (self.dataset_subset_list[i] in dataset_subset) and (self.subject_list[i] in subject) and (
                    self.camera_list[i] in camera) and (self.task_list[i] in task) and (
                    self.hand_list[i] in hand) and (self.object_type_list[i] in object_type):
                valid_indexes.append(i)
        return valid_indexes

    # split data into test and train part
    def split_test_train(vself, valid_indexes, fraction_test):
        test_indexes = random.sample(valid_indexes, int(len(valid_indexes) * fraction_test))
        train_indexes = list(set(valid_indexes) - set(test_indexes))
        return train_indexes, test_indexes

    def switch_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_indexes)
        else:
            return len(self.test_indexes)

    def __getitem__(self, idx):
        if self.mode == 'train':
            sample_index = self.train_indexes[idx]
            if self.mask_type_train == 'Mask':
                mask_path = self.paths_masks[sample_index]
            else:
                mask_path = self.paths_t_masks[sample_index]
        else:
            sample_index = self.test_indexes[idx]

        image_path = self.paths_images[sample_index]
        bbox_path = self.paths_BBox[sample_index]
        sixdof_path = self.paths_6dof[sample_index]
        parameters_path = self.paths_params[sample_index]

        camera_type = self.camera_list[sample_index]
        # bb2d,bb3d,centroid2d,centroid3d,six_dof,internal_calibration_matrix,bb3d_defoult,centroid3d_defoult

        # import internal calibration matrix
        with open(parameters_path) as f:
            defoult_parameters_data = json.load(f)

        internal_calibration_matrix = defoult_parameters_data['K_' + camera_type]
        internal_calibration_matrix = np.reshape(internal_calibration_matrix, (3, 3))

        # import bb2d
        with open(bbox_path) as f:
            bbox_data = json.load(f)

        im_name = self.im_name_list[sample_index]

        bb2d = np.array([list(val.values()) for val in bbox_data[im_name]['BBox_2D_' + camera_type].values()])

        # import six_dof
        trans_vec = np.array(list(bbox_data[im_name]['tracker_to_' + camera_type]['translation'].values()))
        rot_quat = np.array(list(bbox_data[im_name]['tracker_to_' + camera_type]['rotation'].values()))

        # POZOR : from_quat akeceptuje data vo forme x,y,z,w a nase data su w,x,y,z
        rot_quat_xyzw = np.asarray(rot_quat) * 0
        rot_quat_xyzw[0:3] = rot_quat[1:4]
        rot_quat_xyzw[3] = rot_quat[0]

        rot_mat = R.from_quat(rot_quat_xyzw).as_dcm()
        six_dof = np.concatenate((rot_mat, trans_vec[:, np.newaxis]), axis=1)

        # transform six_dof to correct format (is stored as camera to tracker)
        dv = np.zeros((1, 4))
        dv[:, 3] = 1
        six_dof = np.concatenate((six_dof, dv), axis=0)
        six_dof = np.linalg.inv(six_dof)[0:3, :]

        # import bb3d_defoult and centroid3d_defoult
        bb3d_defoult = np.array([list(val.values()) for val in defoult_parameters_data['BB_in_tracker'].values()])
        centroid3d_defoult = np.mean(bb3d_defoult, axis=0)[np.newaxis, :]

        # import centroid2d
        centroid_i = np.c_[centroid3d_defoult, np.ones((centroid3d_defoult.shape[0], 1))].transpose()
        centroid2d, centroid3d = project_from_3d_to_2d(centroid_i, six_dof, internal_calibration_matrix)
        centroid2d = np.rollaxis(centroid2d, axis=1, start=0)
        centroid3d = np.rollaxis(centroid3d, axis=1, start=0)

        # import bb3d
        bb3d_i = np.c_[bb3d_defoult, np.ones((bb3d_defoult.shape[0], 1))].transpose()
        _, bb3d = project_from_3d_to_2d(bb3d_i, six_dof, internal_calibration_matrix)
        bb3d = np.rollaxis(bb3d, axis=1, start=0)

        #  import, scale and randomize image

        img = Image.open(image_path)

        if self.mode == 'train':
            # mask image
            if self.mask_type_train == 'Mask':
                img_m = Image.open(mask_path)
            else:
                # thresholding masks are in grayscale 'L' format, so needs to be converted to 'RGBA'
                img_p = Image.open(mask_path)
                img_m_np = np.array(img_p)
                img_m_np_stacked = np.stack([img_m_np, img_m_np, img_m_np, img_m_np], axis=2)
                img_m = Image.fromarray(img_m_np_stacked, mode='RGBA')

            if self.randomizer_mode == 'bbox':
                img, bb2d, centroid2d = self.randomizer.randomize_bg_bbox_from_mask(img, img_m, six_dof, bb2d,
                                                                                    centroid2d,
                                                                                    self.randomizer_mode_crop)
            elif self.randomizer_mode == 'overlay':
                img, bb2d, centroid2d = self.randomizer.randomize_bg_overlay_from_mask(img, img_m, six_dof, bb2d,
                                                                                       centroid2d,
                                                                                       self.randomizer_mode_crop,
                                                                                       self.randomizer_mode_flip)
            elif self.randomizer_mode == 'overlay_noise_bg':
                img, bb2d, centroid2d = self.randomizer.randomize_bg_overlay_from_mask_noise_bg(img, img_m, six_dof,
                                                                                                bb2d, centroid2d,
                                                                                                self.randomizer_mode_crop,
                                                                                                self.randomizer_mode_flip)
            else:
                pass

        # rescale image
        if self.scale > 1:
            img = img.resize((int(self.max_width / self.scale), int(self.max_height / self.scale)), Image.ANTIALIAS)

        img_np = np.array(img)
        img_np = np.rollaxis(img_np, 2, 0)
        img_np = img_np.astype(np.float32) / 255.

        # create belief maps and affinity maps
        points_belief = []
        one_obj_points = []
        for bb in bb2d:
            one_obj_points.append(tuple(bb))

        # ADD CENTROID
        one_obj_points.append(tuple(centroid2d[0, :]))
        points_belief.append(one_obj_points)
        belief_img = CreateBeliefMap([self.max_width, self.max_height], points_belief, 9, self.sigma, self.gt_scale)
        affinities = GenerateMapAffinity([self.max_width, self.max_height], img.mode, 8, points_belief, centroid2d,
                                         self.gt_scale, self.radius)

        belief_img = belief_img.astype(np.float32) / 255.
        affinities = affinities.astype(np.float32)

        #        info for generating statistics
        batch_label_info = [self.dataset_subset_list[sample_index], self.subject_list[sample_index],
                            self.camera_list[sample_index], self.task_list[sample_index],
                            self.hand_list[sample_index], self.object_type_list[sample_index]]

        batch_file_info = [im_name]

        #        cast data into float32
        bb2d = bb2d.astype(np.float32)
        bb3d = bb3d.astype(np.float32)
        centroid2d = centroid2d.astype(np.float32)
        centroid3d = centroid3d.astype(np.float32)
        six_dof = six_dof.astype(np.float32)

        if self.mode == 'train':
            sample = {'image': img_np, 'belief_img': belief_img, 'affinities': affinities,
                      'bb2d': bb2d, 'bb3d': bb3d, 'centroid2d': centroid2d, 'centroid3d': centroid3d,
                      'six_dof': six_dof,
                      'bb3d_defoult': bb3d_defoult, 'centroid3d_defoult': centroid3d_defoult,
                      'internal_calibration_matrix': internal_calibration_matrix,
                      'batch_label_info': batch_label_info,
                      'batch_file_info': batch_file_info,
                      'DEBUG_train_indexes': np.array(self.train_indexes),
                      'DEBUG_test_indexes': np.array(self.test_indexes),
                      'DEBUG_sample_index': np.array(sample_index)}
            # 'DEBUG_blend_ratio':np.array(blend_ratio)}
        else:
            #  for testing purposes we need an original copy of image, so return that when testing
            img_orig = Image.open(image_path)
            img_orig_np = np.array(img_orig)

            sample = {'image': img_np, 'image_orig': img_orig_np, 'belief_img': belief_img, 'affinities': affinities,
                      'bb2d': bb2d, 'bb3d': bb3d, 'centroid2d': centroid2d, 'centroid3d': centroid3d,
                      'six_dof': six_dof,
                      'bb3d_defoult': bb3d_defoult, 'centroid3d_defoult': centroid3d_defoult,
                      'internal_calibration_matrix': internal_calibration_matrix,
                      'batch_label_info': batch_label_info,
                      'batch_file_info': batch_file_info,
                      'DEBUG_train_indexes': np.array(self.train_indexes),
                      'DEBUG_test_indexes': np.array(self.test_indexes),
                      'DEBUG_sample_index': np.array(sample_index)}
        return sample
