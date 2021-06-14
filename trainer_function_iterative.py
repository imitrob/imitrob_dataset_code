# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:02:39 2021

@author: mattt
"""
import numpy as np
import pickle
import os
import time
import cv2
import copy
import matplotlib.pyplot as plt
from imitrob_dataset_v3 import imitrob_dataset
from torch.utils.data import Dataset, DataLoader
from object_detector import find_objects
from cuboid_PNP_solver import CuboidPNPSolver
from dope_network import dope_net
from error_metrics import rot_trans_err,ADD_error,calculate_AUC

def trainer(params):


    epochs = params['epochs']
    lr = params['lr']
    test_examples_fraction_train = params['test_examples_fraction_train']
    test_examples_fraction_test = params['test_examples_fraction_test']
    batch_size = params['batch_size']
    batch_size_test = params['batch_size_test']
    max_test_batch = params['max_test_batch']
    
    # maximum .jpg visualizations (output examples) to be saved for test
    max_vis = 128
    # test_period = 1000
    
    #NOTE: AUC_test_thresh is used to calculate acc after every epoch, AUC_test_thresh_2 is only for information purpuses
    AUC_test_thresh = 0.02
    AUC_test_thresh_2 = 0.05
    AUC_test_thresh_3 = 0.1
    AUC_test_thresh_range = [0.,0.1]
    
    dataset_type = params['dataset_type']
    
    experiment_name = params['experiment_name']
    
    mask_type = params['mask_type']
    
    crop = params['crop']
    flip = params['flip']
    
    randomizer_mode = params['randomizer_mode']
    
    randomization_prob = params['randomization_prob'] 
    photo_bg_prob = params['photo_bg_prob']
    
    #sets the number of workers for dataloader, set to 0 for windows
    num_workers = params['num_workers']
    
    dataset_path_train = params['dataset_path_train']
    dataset_path_test = params['dataset_path_test']
    
    bg_path = params['bg_path']
    
    
    input_scale = 2
    
    
    sigma = 2
    radius = 2
    
    gpu_device = params['gpu_device']
    
    #test_set_selection = 'fraction'/'subset' if 'fraction' - all types of images are selected for training and
    #random fraction of are selected as a test set, if 'subset' - specify which types of images
    #are selected for train and test, spoecify in lists below
    test_set_selection = params['test_set_selection']
    
    #determine what parts of dataset to include
    
    subject = params['subject']            
    camera = params['camera']
    background = params['background']
    movement_type = params['movement_type']
    movement_direction = params['movement_direction']
    object_type = params['object_type']
    
    subject_test = params['subject_test']            
    camera_test = params['camera_test']
    background_test = params['background_test']
    movement_type_test = params['movement_type_test']
    movement_direction_test = params['movement_direction_test']
    object_type_test = params['object_type_test']
    
    
    attributes_train = [subject,camera,background,movement_type,movement_direction,object_type,mask_type]
    attributes_test = [subject_test,camera_test,background_test,movement_type_test,movement_direction_test,object_type_test,mask_type]
    
    
    #initialize network
    net = dope_net(lr,gpu_device)
    
    #dataset = imitrob_dataset(dataset_path,'train',test_set_selection,test_examples_fraction,[move,subjects,camera,trial],[move_test,subjects_test,camera_test,trial_test]) 
    #dataset = imitrob_dataset(dataset_path,'train',test_set_selection,test_examples_fraction,attributes_train,attributes_test,randomization_prob,input_scale,sigma,radius)
    
    dataset = imitrob_dataset(dataset_path_train,bg_path,'train',test_set_selection,
                              randomizer_mode,mask_type,crop,flip,test_examples_fraction_train,
                              attributes_train,attributes_test,
                              randomization_prob,photo_bg_prob,
                              input_scale,sigma,radius)
    
    dataset_test = imitrob_dataset(dataset_path_test,bg_path,'test',test_set_selection,
                              randomizer_mode,mask_type,False,False,test_examples_fraction_test,
                              attributes_train,attributes_test,
                              0.,0.,
                              input_scale,sigma,radius)
    
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test,shuffle=True, num_workers=num_workers)
    
    
    #create log folder in results folder and write training info into log file
    cwd = os.getcwd()
    
    logs_dir = os.path.join(cwd,'results')
        
    currenttime = time.strftime("%Y_%m_%d___%H_%M_%S")
    logdir = os.path.join(logs_dir, experiment_name)
    os.makedirs(logdir)
    os.chdir(logdir)                        
    f = open('logfile_' + currenttime + '.txt', 'w')
    f.write('dataset type : ' + dataset_type + '\n')
    f.write('test set selection mode : ' + test_set_selection + '\n')
    f.write('attributes train : ' + str(attributes_train) + '\n')
    f.write('attributes test : ' + str(attributes_test) + '\n')
    f.write('randomization_prob  : ' + str(randomization_prob) + '\n')
    f.write('photo_bg_prob  : ' + str(photo_bg_prob) + '\n')
    f.write('randomizer_mode  : ' + randomizer_mode + '\n')
    f.write('mask_type  : ' + mask_type + '\n')
    
    f.write('learning rate : ' + str(lr) + '\n')
    f.write('batch size : ' + str(batch_size) + '\n')
    f.write('test batch size : ' + str(batch_size_test) + '\n')
    f.write('========================================================' + '\n')
    f.write('Training started at:' + (time.strftime("%Y_%m_%d___%H_%M_%S") +'\n'))
    f.write('========================================================' + '\n')
    
    
    #draw one bounding box onto an image
    def draw_box3d(image_file,vertices,centroid,line_thicness,color = (255,0,0)):
        
        vertices = vertices[[1,3,2,0,5,7,6,4],:]
                        
        order = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        
        vertices = vertices[order,:]
        
        for i in range(len(vertices)):
            
            correct_vertex_1 = [vertices[i,0,0],vertices[i,0,1]]
            correct_vertex_2 = [vertices[i,1,0],vertices[i,1,1]]
            
               
            if i<4:
                
                image = cv2.line(image_file, tuple(correct_vertex_1), tuple(correct_vertex_2), color, line_thicness)
                
            else:
                
                image = cv2.line(image_file, tuple(correct_vertex_1), tuple(correct_vertex_2), color, line_thicness)
        
        #    draw centroid circle
                
        image = cv2.circle(image_file,tuple(centroid[0,:]),line_thicness,color,-1)
        
        return image
    
    
    #main train loop
    
    best_acc = 0.
    # log performance parameters after every test
    train_loss_log = []
    test_loss_log = []
    trans_errors_log = []
    rot_errors_log = []
    ADD_errors_log = []
    AUC_acc_log = []
    
    
    timestamp_start = time.time()
    
    for i in range(epochs):
        
        avg_train_loss = []
    
    #    dataset.switch_mode('train')
        
        for train_batch in enumerate(dataloader):
            
            train_images = train_batch[1]['image']
            train_affinities = train_batch[1]['affinities']
            train_beliefs = train_batch[1]['belief_img']
            
            loss = net.train(train_images,train_affinities,train_beliefs)
            
            avg_train_loss.append(loss)
            
    
    #    dataset.switch_mode('test')
    
        test_batch_no = 0
    
        for test_batch in enumerate(dataloader_test):
            
            test_images = test_batch[1]['image']
            test_affinities = test_batch[1]['affinities']
            test_beliefs = test_batch[1]['belief_img']
            bb2d = test_batch[1]['bb2d'].numpy()
            cent2d = test_batch[1]['centroid2d'].numpy()
            RTT_matrix = test_batch[1]['six_dof'].numpy()
            bb3d = test_batch[1]['bb3d'].numpy()
            centroid3d = test_batch[1]['centroid3d'].numpy()
            
    #        defoult parameters, can be different for different objects
            bb3d_defoult = test_batch[1]['bb3d_defoult'].numpy()
            centroid3d_defoult = test_batch[1]['centroid3d_defoult'].numpy()
            internal_calibration_matrix = test_batch[1]['internal_calibration_matrix'].numpy()
            
            images = test_batch[1]['image_orig'].numpy()
            
                #        run test images through network
            belief,affinity,loss = net.test(test_images,test_affinities,test_beliefs)
            
        #        clip the range of output belief images between 0 and 1, slightly helps the performance of object_detector
            belief = belief.clip(min=0.,max=1.)
            
            if test_batch_no == 0:
                
                belief_test = copy.deepcopy(belief)
                affinity_test = copy.deepcopy(affinity)
                loss_test = loss
                
                original_images = copy.deepcopy(images)
                bb2d_gt = copy.deepcopy(bb2d)
                cent2d_gt = copy.deepcopy(cent2d)
                RTT_matrix_gt = copy.deepcopy(RTT_matrix)
                bb3d_gt = copy.deepcopy(bb3d)
                centroid3dgt = copy.deepcopy(centroid3d)
                
                bb3d_defoult_gt = copy.deepcopy(bb3d_defoult)
                centroid3d_defoult_gt = copy.deepcopy(centroid3d_defoult)
                internal_calibration_matrix_gt = copy.deepcopy(internal_calibration_matrix)
                
            else:
                
                belief_test = np.concatenate((belief_test,belief),axis=0)
                affinity_test = np.concatenate((affinity_test,affinity),axis=0)
                loss_test += loss
                
                original_images = np.concatenate((original_images,images),axis=0)
                bb2d_gt = np.concatenate((bb2d_gt,bb2d),axis=0)
                cent2d_gt = np.concatenate((cent2d_gt,cent2d),axis=0)
                RTT_matrix_gt = np.concatenate((RTT_matrix_gt,RTT_matrix),axis=0)
                bb3d_gt = np.concatenate((bb3d_gt,bb3d),axis=0)
                centroid3dgt = np.concatenate((centroid3dgt,centroid3d),axis=0)
                
                bb3d_defoult_gt = np.concatenate((bb3d_defoult_gt,bb3d_defoult),axis=0)
                centroid3d_defoult_gt = np.concatenate((centroid3d_defoult_gt,centroid3d_defoult),axis=0)
                internal_calibration_matrix_gt = np.concatenate((internal_calibration_matrix_gt,internal_calibration_matrix),axis=0)
                
            
            if test_batch_no == (max_test_batch-1):
                
                loss_test = loss_test/max_test_batch
                
                break
            
            test_batch_no += 1
            
    
        
        object_finder_errors = 0
        object_finder_exceptions = 0
        
    #        make dir inside logdir where bb visualizations will be stored
        ep_sample_dir = os.path.join(logdir,'Samples_episode_' + str(i))
        
        os.makedirs(ep_sample_dir)
        
    #        lists for calculating avg translation and rotation errors
        translation_err_list = []
        rotation_err_list = []
        ADD_err_list = []
        AUC_err_list = []
        
    #        try to find objects form the output of the network
        for j in range(len(belief_test)):
            
    #            try to find objects using the network output
    #            object finder can in small amount of cases throw an exception: IndexError: index 30 is out of bounds for axis 1 with size 30
    #            so far dont know the cause of this error, most probably an bug in the original implementation
    
            cuboid2d,_ = find_objects(belief_test[j,:,:,:].astype(np.float64),affinity_test[j,:,:,:].astype(np.float64),1,input_scale)
            
            
            if cuboid2d is None:
                
                object_finder_errors += 1
                
                translation_err_list.append(10000.)
                rotation_err_list.append(10000.)
                ADD_err_list.append(10000.)
                
                continue
            
            else:
            
    #                the data in cuboid2d is in network output resolution (30x40) not the original resolution of data (480x640)
    #                the input to the network is is subsampled to 480/2,640/2 = 240,320
    #                the network then subsamples the input by 8 until it reaches the output
    #                therefore we need to scale the coordinates of bb by 8*2 = 16 if we want to get the coordinates in original input space
    #            cuboid2d = cuboid2d*8*input_scale
                
    #                use pnp soilver to get the pose
    #                location: [x,y,z]
    #                quaternion: [x,y,z,w]
    #                projected_points (9x2) (DONT USE THIS, sometimes gives incorrect output)
    #                RTT_matrix (3x4)
                
    #           cv2.solvePnPRansac gives this error on some ocasions: 
    #             File "E:\Python 3 projects\learning_task_from_rgbd_cameras_and_language\DOPE_imitrob_v4\cuboid_PNP_solver.py", line 160, in solve_pnp
    #             rt_matrix = quaternion.matrix33
    #
    #                AttributeError: 'NoneType' object has no attribute 'matrix33'
                
                vertices = np.concatenate((bb3d_defoult_gt[j,:,:],centroid3d_defoult_gt[j,:,:]))
                
                pnp_solver = CuboidPNPSolver(camera_intrinsic_matrix = internal_calibration_matrix_gt[j,:,:],
                                 cuboid3d = vertices)
                
                location, quaternion, projected_points,RTT_matrix = pnp_solver.solve_pnp(cuboid2d)
                
                
                bb_approximation = projected_points[0:8,:]
                cent_approximation = projected_points[8:9,:]
                
                img = original_images[j,:,:,:]             
                
    #                draw gt bb
                annotation_image = draw_box3d(img,bb2d_gt[j,:,:].astype(np.float32),cent2d_gt[j,:,:].astype(np.float32),5,(0,255,0))
    #                draw predicted bb
                annotation_image = draw_box3d(annotation_image,bb_approximation.astype(np.float32),cent_approximation.astype(np.float32),5)
                
                if j < max_vis:
    
    #                save visualization
                    cv2.imwrite(os.path.join(ep_sample_dir,'sample_' + str(j) + '.jpg'),cv2.cvtColor(annotation_image, cv2.COLOR_RGB2BGR))
                
    #                calculate translation and rotation err
                translation_err,rotation_err = rot_trans_err(RTT_matrix_gt[j,:,:],RTT_matrix)
    #                calculate ADD error
                ADD_err = ADD_error(bb3d_gt[j,:,:],centroid3dgt[j,:,:],RTT_matrix,bb3d_defoult_gt[j,:,:],centroid3d_defoult_gt[j,:,:])
                
                translation_err_list.append(translation_err)
                rotation_err_list.append(rotation_err)
                ADD_err_list.append(ADD_err)
                
        
    
        AUC_acc = calculate_AUC(ADD_err_list,AUC_test_thresh)        
                
        f.write('Object finder errors at epoch : ' + str(i) + ' : ' + str(object_finder_errors) +
                ' Object finder exceptions at epoch : ' + str(i) + ' : ' + str(object_finder_exceptions) +
                ' train loss : ' + str(sum(avg_train_loss)/len(avg_train_loss)) + 
                ' test loss : ' + str(loss_test) + 
                ' translation_err : ' + str(sum(translation_err_list)/len(translation_err_list))+ 
                ' rotation_err : ' + str(sum(rotation_err_list)/len(rotation_err_list)) + 
                ' ADD_err : ' + str(sum(ADD_err_list)/len(ADD_err_list)) + 
                ' AUC_acc : ' + str(AUC_acc) + '\n')
        
        train_loss_log.append(sum(avg_train_loss)/len(avg_train_loss))
        test_loss_log.append(loss_test)
        trans_errors_log.append(sum(translation_err_list)/len(translation_err_list))
        rot_errors_log.append(sum(rotation_err_list)/len(rotation_err_list))
        ADD_errors_log.append(sum(ADD_err_list)/len(ADD_err_list))
        AUC_acc_log.append(AUC_acc)
                
        f.flush()
        
    #        if test loss improved comapred to last test, save the current weights of the model
        if AUC_acc >= best_acc:
            
            net.save_model(os.path.join(logdir,'checkpoint.pth.tar'))
            best_acc = AUC_acc
            max_acc_episode = i
                
        print('Epoch : ' + str(i) + ' acc : ' + str(AUC_acc))
    
    
    #RUN FINAL TEST WITH BEST WEIGHTS
    #================================
    
    net.load_model(os.path.join(logdir,'checkpoint.pth.tar'))
    
    object_finder_errors_test_final = 0
    translation_err_list_final = []
    rotation_err_list_final = []
    ADD_err_list_final = []
    
    projected_points_buffer = []
    cuboid2d_buffer = []
    RTT_matrix_buffer = []
    location_buffer = []
    quaternion_buffer = []
    
    RTT_matrix_gt_buffer = []
    bb3d_gt_buffer = []
    
    info_buffer = []
    file_buffer = []
    
    
    #dataset.switch_mode('test')
    
    for test_batch in enumerate(dataloader_test):
            
        test_images = test_batch[1]['image']
        test_affinities = test_batch[1]['affinities']
        test_beliefs = test_batch[1]['belief_img']
        bb2d_gt = test_batch[1]['bb2d'].numpy()
        cent2d_gt = test_batch[1]['centroid2d'].numpy()
        RTT_matrix_gt = test_batch[1]['six_dof'].numpy()
        bb3d_gt = test_batch[1]['bb3d'].numpy()
        centroid3dgt = test_batch[1]['centroid3d'].numpy()
        batch_label_info = test_batch[1]['batch_label_info']
        file_info = test_batch[1]['batch_file_info']
        
        bb3d_defoult = test_batch[1]['bb3d_defoult'].numpy()
        centroid3d_defoult = test_batch[1]['centroid3d_defoult'].numpy()
        internal_calibration_matrix = test_batch[1]['internal_calibration_matrix'].numpy()
        
        original_images = test_batch[1]['image_orig'].numpy()
    
        belief_test,affinity_test,loss_test = net.test(test_images,test_affinities,test_beliefs)
    
        belief_test = belief_test.clip(min=0.,max=1.)
        
        for j in range(len(belief_test)):
            
            cuboid2d,_ = find_objects(belief_test[j,:,:,:].astype(np.float64),affinity_test[j,:,:,:].astype(np.float64),1,input_scale)
            
            RTT_matrix_gt_buffer.append(RTT_matrix_gt[j,:,:])
            bb3d_gt_buffer.append(bb3d_gt[j,:,:])
            
            info_buffer.append([batch_label_info[0][j],batch_label_info[1][j],batch_label_info[2][j],batch_label_info[3][j]])
            file_buffer.append(file_info[0][j])
            
            if cuboid2d is None:
                
                object_finder_errors_test_final += 1
                translation_err_list_final.append(10000.)
                rotation_err_list_final.append(10000.)
                ADD_err_list_final.append(10000.)
                cuboid2d_buffer.append('NA')
                projected_points_buffer.append('NA')
                RTT_matrix_buffer.append('NA')
                location_buffer.append('NA')
                quaternion_buffer.append('NA')
                
                continue
            
            else:
                
                vertices = np.concatenate((bb3d_defoult[j,:,:],centroid3d_defoult[j,:,:]))
                
                pnp_solver = CuboidPNPSolver(camera_intrinsic_matrix = internal_calibration_matrix[j,:,:],
                                 cuboid3d = vertices)
                
                location, quaternion, projected_points,RTT_matrix = pnp_solver.solve_pnp(cuboid2d)
                
                translation_err,rotation_err = rot_trans_err(RTT_matrix_gt[j,:,:],RTT_matrix)
                ADD_err = ADD_error(bb3d_gt[j,:,:],centroid3dgt[j,:,:],RTT_matrix,bb3d_defoult[j,:,:],centroid3d_defoult[j,:,:])
                
                translation_err_list_final.append(translation_err)
                rotation_err_list_final.append(rotation_err)
                ADD_err_list_final.append(ADD_err)
                
                projected_points_buffer.append(projected_points)
                cuboid2d_buffer.append(cuboid2d)
                RTT_matrix_buffer.append(RTT_matrix)
                location_buffer.append(location)
                quaternion_buffer.append(quaternion)
    
    
    AUC_acc_final = calculate_AUC(ADD_err_list_final,AUC_test_thresh)
    AUC_acc_final_2 = calculate_AUC(ADD_err_list_final,AUC_test_thresh_2)
    AUC_acc_final_3 = calculate_AUC(ADD_err_list_final,AUC_test_thresh_3)
    
    f.write('                                              ' + '\n')
    f.write('                                              ' + '\n')
    f.write('Best performance was : ' + str(best_acc) + ' at epoch : ' + str(max_acc_episode) + '\n')
    
    f.write('                                              ' + '\n')
    f.write('                                              ' + '\n')
    
    f.write('                   FINAL PERFORMANCE                           ' + '\n')
    
    f.write(' translation_err_final : ' + str(sum(translation_err_list_final)/len(translation_err_list_final))+ 
                ' rotation_err_final : ' + str(sum(rotation_err_list_final)/len(rotation_err_list_final)) + 
                ' ADD_err_final : ' + str(sum(ADD_err_list_final)/len(ADD_err_list_final)) +
                ' object_finder_errors_test_final : ' + str(object_finder_errors_test_final) + ' | ' + str(len(dataset)) + 
                ' AUC_acc_final at threshold ' + str(AUC_test_thresh) + ' : ' + str(AUC_acc_final) + 
                ' AUC_acc_final at threshold ' + str(AUC_test_thresh_2) + ' : ' + str(AUC_acc_final_2) +
                ' AUC_acc_final at threshold ' + str(AUC_test_thresh_3) + ' : ' + str(AUC_acc_final_3)  + '\n')
    
    f.write('                   FINAL PERFORMANCE                           ' + '\n')
    
    timestamp_stop = time.time()
    
    f.write('                                              ' + '\n')
    f.write('                                              ' + '\n')
    
    f.write('Execution time was : ' + str((timestamp_stop-timestamp_start)/3600.) + ' hours')
            
    f.close()
    
    
    #here we generate AUC curve from: https://arxiv.org/pdf/1809.10790.pdf
    
    thresh = []
    acc_at_thresh = []
    
    for i in range(0,1001):
        
        tr = AUC_test_thresh_range[0] + (AUC_test_thresh_range[1]/1000)*i
        AUC = calculate_AUC(ADD_err_list_final,tr)
        
        thresh.append(tr)
        acc_at_thresh.append(AUC)
      
      
    plt.plot(thresh, acc_at_thresh) 
    
    plt.ylim((0,1))  
     
    plt.xlabel('Average distance threshold in meters') 
     
    plt.ylabel('ADD pass rate') 
      
    # giving a title to my graph 
    plt.title(currenttime + '_' + 'DOPE')
    
    plt.savefig(os.path.join(logdir,'AUC_curve.png'),dpi = 600)  
      
    # function to show the plot 
    #plt.show()
    
#    clear current figure
    plt.close()
    
    
    
    #save error metrics file to log folder in pickle file
    #err_metrics = [train_loss_log,test_loss_log,trans_errors_log,rot_errors_log,ADD_errors_log,AUC_acc_log,thresh,acc_at_thresh]
    err_metrics = {'AUC_acc_final':AUC_acc_final,
                   'translation_err_list_final':translation_err_list_final,
                   'rotation_err_list_final':rotation_err_list_final,
                   'ADD_err_list_final':ADD_err_list_final,
                   'thresh':thresh,
                   'acc_at_thresh':acc_at_thresh,
                   'train_loss_log':train_loss_log,
                   'test_loss_log':test_loss_log,
                   'trans_errors_log':trans_errors_log,
                   'rot_errors_log':rot_errors_log,
                   'ADD_errors_log':ADD_errors_log,
                   'AUC_acc_log':AUC_acc_log,
                   'object_finder_errors_test_final':object_finder_errors_test_final,
                   'cuboid_2d_buffer':cuboid2d_buffer,
                   'cuboid_3d_buffer':projected_points_buffer,
                   'RTT_matrix_buffer':RTT_matrix_buffer,
                   'location_buffer':location_buffer,
                   'quaternion_buffer':quaternion_buffer,
                   'RTT_matrix_gt_buffer':RTT_matrix_gt_buffer,
                   'bb3d_gt_buffer':bb3d_gt_buffer,
                   'info_buffer': info_buffer,
                   'file_buffer' : file_buffer}
        
    with open('err_metrics.pkl', 'wb') as f:
               pickle.dump(err_metrics, f)
               
               
    os.chdir(cwd)
    
#    net.empty_cuda_cache()