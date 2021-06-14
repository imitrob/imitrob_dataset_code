# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:37:34 2021

@author: mattt
"""

from trainer_function_iterative import trainer

dataset_path_train = ''
dataset_path_test = ''
bg_path = ''
gpu_device = 0
num_workers = 8


exp_same_subject_1_wo_aug = {'epochs':10,
          'lr':0.0001,
          'test_examples_fraction_train':0.,
          'test_examples_fraction_test':1.,
          'batch_size':16,
          'batch_size_test':16,
          'max_test_batch':64,
          'dataset_type':'groutfloat',
          'experiment_name':'groutfloat_same_subject_S1_wo_aug',
          'mask_type':'Mask',
          'crop':True,
          'flip':True,
          'randomizer_mode':'none',
          'randomization_prob':0.75,
          'photo_bg_prob':1.,
          'num_workers':num_workers,
          'dataset_path_train':dataset_path_train,
          'dataset_path_test':dataset_path_test,
          'bg_path':bg_path,
          'gpu_device':gpu_device,
          'test_set_selection':'subset',
          'subject':['S1'],
          'camera':['C1','C2'],
          'background':['green'],
          'movement_type':['random','round','sweep','press','frame','sparsewave','densewave'],
          'movement_direction':['left','right'],
          'object_type':['groutfloat'],
          'subject_test':['S1'],
          'camera_test':['C1','C2'],
          'background_test':['table','normalclutter','superclutter'],
          'movement_type_test':['random','round','sweep','press','frame','sparsewave','densewave'],
          'movement_direction_test':['left','right'],
          'object_type_test':['groutfloat']}


experiments_list = []


for experiment in experiments_list:
    
    trainer(experiment)
    
    print('===========================================')
    print('Experiment ' + experiment['experiment_name'] + ' FINISHED!')
    print('===========================================')