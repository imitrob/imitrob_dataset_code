# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 17:08:50 2021

@author: mattt
"""

import numpy as np
import pickle
import os
import time
import cv2
import copy
import argparse
import matplotlib.pyplot as plt
from imitrob_dataset import imitrob_dataset
from torch.utils.data import Dataset, DataLoader
from object_detector import find_objects
from cuboid_PNP_solver import CuboidPNPSolver
from dope_network import dope_net
from error_metrics import rot_trans_err,ADD_error,calculate_AUC

parser = argparse.ArgumentParser(description='Compute Accuracy')
parser.add_argument('--results', type=str, default='', metavar='E',
                    help='Path to the directory with err_metrics.pkl')
parser.add_argument('--target', type=str, default='', metavar='E',
                    help='Path to the target results file')
args = parser.parse_args()

select_subset = False
subjects = ['S1','S2','S3','S4']
cameras = ['C1','C2']

results_location = args.results

target_results_file = args.target

res_text = open(target_results_file,'w')

res_dirs = os.listdir(results_location)

for rd in res_dirs:
    
    res_path = os.path.join(os.path.join(results_location,rd),'err_metrics.pkl')
    
    try:
    
        with open(res_path,'rb') as file: data = pickle.load(file)
        
    except:
        
        continue
    
    ADD_err_list_final = data['ADD_err_list_final']
    info_buffer = data['info_buffer']
    
    if select_subset:
    
        ADD_errors_selection = []
        
        for i in range(len(ADD_err_list_final)):
            
            info = info_buffer[i]
            
            if (info[0] in subjects) and (info[1] in cameras):
                
                ADD_errors_selection.append(ADD_err_list_final[i])
                
    else:
        
        ADD_errors_selection = data['ADD_err_list_final']
            
            
    
    AUC_acc_2cm = calculate_AUC(ADD_errors_selection,0.02)
    AUC_acc_5cm = calculate_AUC(ADD_errors_selection,0.05)
    AUC_acc_10cm = calculate_AUC(ADD_errors_selection,0.1)
    
    AUC_acc_2cm = str(AUC_acc_2cm*100).split('.')
    AUC_acc_2cm = AUC_acc_2cm[0] + ',' + AUC_acc_2cm[1]
    
    AUC_acc_5cm = str(AUC_acc_5cm*100).split('.')
    AUC_acc_5cm = AUC_acc_5cm[0] + ',' + AUC_acc_5cm[1]
    
    AUC_acc_10cm = str(AUC_acc_10cm*100).split('.')
    AUC_acc_10cm = AUC_acc_10cm[0] + ',' + AUC_acc_10cm[1]
    
    res_text.write(rd + '  : ' + ' ADD 2cm : ' + AUC_acc_2cm + ' , ADD 5cm : ' + AUC_acc_5cm + ' , ADD 10cm : ' + AUC_acc_10cm+ '\n')
    res_text.write('======================================================================================================================' + '\n')
    
res_text.close()
