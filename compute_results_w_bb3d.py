# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:50:01 2021

@author: mattt
"""

import numpy as np
import math
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from imitrob_dataset import imitrob_dataset
import argparse


parser = argparse.ArgumentParser(description='Compute Accuracy')
parser.add_argument('--results-path', type=str, default='',
                    help='')
parser.add_argument('--rec-metrics', type=str, default='',
                    help='')
parser.add_argument('--test-data', type=str, default='',
                    help='Path to dataset test data')
parser.add_argument('--bg-path', type=str, default="",
                    help='Path to the backgrounds folder')
parser.add_argument('--tool', type=str, default="gluegun",
                    help='Which tool to evaluate')
args = parser.parse_args()


results_path = args.results_path

recomputed_metrics_dir = args.rec_metrics
recomputed_metrics_file_name = 'err_metrics_gluegun_test_BgBlend.pkl'

dataset_path_test = args.test_data
bg_path = args.bg_path
test_set_selection = 'subset'
randomizer_mode = 'none'
mask_type = 'Mask'
test_examples_fraction_test = 1.
dataset_type = 'gluegun'
batch_size_test = 1
num_workers = 0

# original images have size (480x640), network input has size (480/2,640/2)
input_scale = 1

#use sigma = 1 and radius = 1 for 320x240, use sigma = 2 and radius = 3 for 640x480
sigma = 4
radius = 4


subject = ['S1','S2','S3','S4']
camera = ['C1','C2']
background = ['green']
movement_type = ['random','round','sweep','press']
movement_direction = ['left','right']
object_type = [dataset_type]

subject_test = ['S1','S2','S3','S4']
camera_test = ['C1','C2']
background_test = ['table']
movement_type_test = ['sparsewave']
movement_direction_test = ['left','right']
object_type_test = [dataset_type]

attributes_train = [subject,camera,background,movement_type,movement_direction,object_type,mask_type]
attributes_test = [subject_test,camera_test,background_test,movement_type_test,movement_direction_test,object_type_test,mask_type]

dataset_test = imitrob_dataset(dataset_path_test,bg_path,'test',test_set_selection,
                              randomizer_mode,mask_type,False,False,test_examples_fraction_test,
                              attributes_train,attributes_test,
                              0.,0.,
                              input_scale,sigma,radius)


dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test,shuffle=True, num_workers=num_workers)

# aaa = []
# bbb = []

for test_batch in enumerate(dataloader_test):

    bb3d_defoult = test_batch[1]['bb3d_default'].numpy()
    bb3d_defoult = bb3d_defoult[0,:,:]
    centroid3d_defoult = test_batch[1]['centroid3d_default'].numpy()

    # aaa.append(bb3d_defoult)
    # bbb.append(centroid3d_defoult)

    break


with open(results_path, 'rb') as f:

    results = pickle.load(f)

f.close()

bb3d_gt_buffer = results['bb3d_gt_buffer']
RTT_matrix_buffer = results['RTT_matrix_buffer']
RTT_matrix_gt_buffer = results['RTT_matrix_gt_buffer']

bb3d_prediction_buffer = []

for i in range(len(bb3d_gt_buffer)):

    RTT_matrix_pred = RTT_matrix_buffer[i]

    if type(RTT_matrix_pred) != str:

        bb3d_defoult_i = np.c_[bb3d_defoult, np.ones((bb3d_defoult.shape[0], 1))].transpose()

        points3d_pred = RTT_matrix_pred.dot(bb3d_defoult_i)

        points3d_pred = np.rollaxis(points3d_pred,1,0)

        bb3d_prediction_buffer.append(points3d_pred)

    else:

        bb3d_prediction_buffer.append(RTT_matrix_pred)


results['bb3d_prediction_buffer'] = bb3d_prediction_buffer

# res_path = os.path.join(recomputed_metrics_dir,recomputed_metrics_file_name)

# with open(res_path, 'wb') as f:
#      pickle.dump(results, f)

# f.close()


# alternative err metric
def alt_rot_met(rot_pred,rot_gt):

    err = np.arccos((np.trace(np.dot(np.linalg.inv(rot_pred),rot_gt))-1)/2)
    err = math.degrees(err)

    return err

translation_err_list_final = results['translation_err_list_final']
rotation_err_list_final = results['rotation_err_list_final']

rot_sum = 0
rot_alt_sum = 0
trans_sum = 0
count = 0

rot_list = []
rot_alt_list = []

for i in range(len(translation_err_list_final)):

    rot = rotation_err_list_final[i]
    trans = translation_err_list_final[i]

    # alt metric
    if RTT_matrix_buffer[i] != 'NA':

        rot_alt = alt_rot_met(RTT_matrix_buffer[i][:,0:3],RTT_matrix_gt_buffer[i][:,0:3])

    else:

        rot_alt = 10000

    if (rot < 360) and (trans < 1):

        if rot < 180:

            rot_sum += rot
            rot_list.append(rot)

        else:

            rot_sum += (360-rot)
            rot_list.append(360-rot)

        trans_sum += trans

        count += 1



    if (rot_alt < 360) and (trans < 1):

        if rot_alt < 180:

            rot_alt_sum += rot_alt
            rot_alt_list.append(rot_alt)

        else:

            rot_alt_sum += (360-rot_alt)
            rot_alt_list.append(360-rot_alt)



print('valid prediction percentage : ' + str((count/len(translation_err_list_final))*100) + '; avg translation error : ' + str(trans_sum/count) + '; avg rotation error : ' + str(rot_sum/count) + '; avg rotation error alternative : ' + str(rot_alt_sum/count))

