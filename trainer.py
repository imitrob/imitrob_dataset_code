# -*- coding: utf-8 -*-
"""
    6D estimator trainer script
    Copyright (C) 2020  Matus Tuna

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
import numpy as np
import os
import argparse
import time
import cv2
import copy
from imitrob_dataset import imitrob_dataset
from torch.utils.data import DataLoader
from object_detector import find_objects
from cuboid_PNP_solver import CuboidPNPSolver
from dope_network import dope_net
from error_metrics import rot_trans_err, ADD_error, calculate_AUC
from evaluation import main as test_model
from evaluation import draw_box3d

def train():
    # main train loop
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

        for train_batch in enumerate(dataloader):
            train_images = train_batch[1]['image']
            train_affinities = train_batch[1]['affinities']
            train_beliefs = train_batch[1]['belief_img']

            loss = net.train(train_images, train_affinities, train_beliefs)

            avg_train_loss.append(loss)

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

            # default parameters, can be different for different objects
            bb3d_default = test_batch[1]['bb3d_default'].numpy()
            centroid3d_default = test_batch[1]['centroid3d_default'].numpy()
            internal_calibration_matrix = test_batch[1]['internal_calibration_matrix'].numpy()

            images = test_batch[1]['image_orig'].numpy()

            # run test images through network
            belief, affinity, loss = net.test(test_images, test_affinities, test_beliefs)

            # clip the range of output belief images between 0 and 1, slightly helps the performance of object_detector
            belief = belief.clip(min=0., max=1.)

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

                bb3d_default_gt = copy.deepcopy(bb3d_default)
                centroid3d_default_gt = copy.deepcopy(centroid3d_default)
                internal_calibration_matrix_gt = copy.deepcopy(internal_calibration_matrix)

            else:

                belief_test = np.concatenate((belief_test, belief), axis=0)
                affinity_test = np.concatenate((affinity_test, affinity), axis=0)
                loss_test += loss

                original_images = np.concatenate((original_images, images), axis=0)
                bb2d_gt = np.concatenate((bb2d_gt, bb2d), axis=0)
                cent2d_gt = np.concatenate((cent2d_gt, cent2d), axis=0)
                RTT_matrix_gt = np.concatenate((RTT_matrix_gt, RTT_matrix), axis=0)
                bb3d_gt = np.concatenate((bb3d_gt, bb3d), axis=0)
                centroid3dgt = np.concatenate((centroid3dgt, centroid3d), axis=0)

                bb3d_default_gt = np.concatenate((bb3d_default_gt, bb3d_default), axis=0)
                centroid3d_default_gt = np.concatenate((centroid3d_default_gt, centroid3d_default), axis=0)
                internal_calibration_matrix_gt = np.concatenate(
                    (internal_calibration_matrix_gt, internal_calibration_matrix), axis=0)

            if test_batch_no == (max_test_batch - 1):
                loss_test = loss_test / max_test_batch

                break

            test_batch_no += 1

        object_finder_errors = 0
        object_finder_exceptions = 0

        # make dir inside logdir where bb visualizations will be stored
        ep_sample_dir = os.path.join(logdir, 'Samples_episode_' + str(i))

        os.makedirs(ep_sample_dir)

        # lists for calculating avg translation and rotation errors
        translation_err_list = []
        rotation_err_list = []
        ADD_err_list = []
        AUC_err_list = []

        # try to find objects form the output of the network
        for j in range(len(belief_test)):

            # try to find objects using the network output
            # object finder can in small amount of cases throw an exception: IndexError: index 30 is out of bounds for axis 1 with size 30
            # so far dont know the cause of this error, most probably an bug in the original implementation

            cuboid2d, _ = find_objects(belief_test[j, :, :, :].astype(np.float64),
                                       affinity_test[j, :, :, :].astype(np.float64), 1, input_scale)

            if cuboid2d is None:

                object_finder_errors += 1

                translation_err_list.append(10000.)
                rotation_err_list.append(10000.)
                ADD_err_list.append(10000.)

                continue

            else:
                ''' The data in cuboid2d is in network output resolution (30x40), not the original resolution of data (480x640)
                 The input to the network is is subsampled to 480/2,640/2 = 240,320, the network then subsamples the input
                 by 8, until it reaches the output.Therefore ,we need to scale the coordinates of bb by 8*2 = 16 if we want
                 to get the coordinates in the original input space.
                 cuboid2d = cuboid2d*8*input_scale

                 use pnp soilver to get the pose
                 location: [x,y,z]
                 quaternion: [x,y,z,w]
                 projected_points (9x2) (DONT USE THIS, sometimes gives incorrect output)
                 RTT_matrix (3x4)'''

                vertices = np.concatenate((bb3d_default_gt[j, :, :], centroid3d_default_gt[j, :, :]))

                pnp_solver = CuboidPNPSolver(camera_intrinsic_matrix=internal_calibration_matrix_gt[j, :, :],
                                             cuboid3d=vertices)

                location, quaternion, projected_points, RTT_matrix = pnp_solver.solve_pnp(cuboid2d)

                bb_approximation = projected_points[0:8, :]
                cent_approximation = projected_points[8:9, :]

                img = original_images[j, :, :, :]

                # draw gt bb
                annotation_image = draw_box3d(img, bb2d_gt[j, :, :].astype(np.float32),
                                              cent2d_gt[j, :, :].astype(np.float32), 5, (0, 255, 0))
                # draw predicted bb
                annotation_image = draw_box3d(annotation_image, bb_approximation.astype(np.float32),
                                              cent_approximation.astype(np.float32), 5)

                if j < max_vis:
                    # save visualization
                    cv2.imwrite(os.path.join(ep_sample_dir, 'sample_' + str(j) + '.jpg'),
                                cv2.cvtColor(annotation_image, cv2.COLOR_RGB2BGR))

                # calculate translation and rotation err
                translation_err, rotation_err = rot_trans_err(RTT_matrix_gt[j, :, :], RTT_matrix)
                # calculate ADD error
                ADD_err = ADD_error(bb3d_gt[j, :, :], centroid3dgt[j, :, :], RTT_matrix, bb3d_default_gt[j, :, :],
                                    centroid3d_default_gt[j, :, :])

                translation_err_list.append(translation_err)
                rotation_err_list.append(rotation_err)
                ADD_err_list.append(ADD_err)

        AUC_acc = calculate_AUC(ADD_err_list, AUC_test_thresh)

        f.write('Object finder errors at epoch : ' + str(i) + ' : ' + str(object_finder_errors) +
                ' Object finder exceptions at epoch : ' + str(i) + ' : ' + str(object_finder_exceptions) +
                ' train loss : ' + str(sum(avg_train_loss) / len(avg_train_loss)) +
                ' test loss : ' + str(loss_test) +
                ' translation_err : ' + str(sum(translation_err_list) / len(translation_err_list)) +
                ' rotation_err : ' + str(sum(rotation_err_list) / len(rotation_err_list)) +
                ' ADD_err : ' + str(sum(ADD_err_list) / len(ADD_err_list)) +
                ' AUC_acc : ' + str(AUC_acc) + '\n')

        train_loss_log.append(sum(avg_train_loss) / len(avg_train_loss))
        test_loss_log.append(loss_test)
        trans_errors_log.append(sum(translation_err_list) / len(translation_err_list))
        rot_errors_log.append(sum(rotation_err_list) / len(rotation_err_list))
        ADD_errors_log.append(sum(ADD_err_list) / len(ADD_err_list))
        AUC_acc_log.append(AUC_acc)

        f.flush()

        # if test loss improved comapred to last test, save the current weights of the model
        if AUC_acc >= best_acc:
            net.save_model(os.path.join(logdir, 'checkpoint.pth.tar'))
            best_acc = AUC_acc
            max_acc_episode = i

        print('Epoch : ' + str(i) + ' acc : ' + str(AUC_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Accuracy')
    parser.add_argument('--traindata', type=str, default='',
                        help='Path to the Train directory of Imitrob')
    parser.add_argument('--testdata', type=str, default='',
                        help='Path to the Test directory of Imitrob')
    parser.add_argument('--bg_path', type=str, default="",
                        help='Path to the backgrounds folder')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--batch_size_test', type=int, default=128,
                        help='Batch size for testing')
    parser.add_argument('--max_vis', type=int, default=128,
                        help='Maximum .jpg visualizations (output examples) to be saved for test')
    parser.add_argument('--exp_name', type=str, default="",
                        help='Name of the folder were results will be stored. Choose unique name for every experiment')
    parser.add_argument('--mask_type', action='store', choices=['Mask', 'Mask_thresholding'], type=str, default='Mask',
                        help='Choose the type of mask used during training. Mask or Mask_thresholding')
    parser.add_argument('--randomizer_mode', action='store', choices=['none', 'bbox', 'overlay', 'overlay_noise_bg'],
                        type=str, default='overlay',
                        help='Choose the type of input augmentation. none,bbox,overlay or overlay_noise_bg')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of dataloader workers. Use 0 for Win')
    parser.add_argument('--gpu_device', type=int, default=0,
                        help='Gpu device to use for training')
    parser.add_argument('--dataset_type', action='store', choices=['gluegun', 'groutfloat', 'roller'], type=str,
                        default='gluegun',
                        help='Choose the type of data used in training and testing. gluegun, groutfloat or roller')
    parser.add_argument('--subject', type=str, default="",
                        help='List of subjects to be used for training. All subjects: S1,S2,S3,S4')
    parser.add_argument('--camera', type=str, default="",
                        help='List of cameras to be used for training. All cameras: C1,C2')
    parser.add_argument('--hand', type=str, default="",
                        help='List of hands to be used for training. All hands: LH,RH')
    parser.add_argument('--subject_test', type=str, default="",
                        help='List of subjects to be used for testing. All subjects: S1,S2,S3,S4')
    parser.add_argument('--camera_test', type=str, default="",
                        help='List of cameras to be used for testing. All cameras: C1,C2')
    parser.add_argument('--hand_test', type=str, default="",
                        help='List of hands to be used for testing. All hands: LH,RH')
    parser.add_argument('--task_test', type=str, default="",
                        help='List of tasks to be used for testing. All tasks: clutter,round,sweep,press,frame,sparsewave,densewave')
    parser.add_argument('--skip_testing', default=False, action='store_true',
                        help='If this flag is used, the testing after training will be omitted.')
    args = parser.parse_args()

    epochs = args.epochs
    lr = args.lr
    test_examples_fraction_train = 0.
    test_examples_fraction_test = 1.
    batch_size = args.batch_size
    batch_size_test = args.batch_size_test
    max_test_batch = 8

    # maximum .jpg visualizations (output examples) to be saved for test
    max_vis = args.max_vis
    # test_period = 1000

    # NOTE: AUC_test_thresh is used to calculate acc after every epoch, AUC_test_thresh_2 is only for information purpuses
    AUC_test_thresh = 0.02
    AUC_test_thresh_2 = 0.1
    AUC_test_thresh_range = [0., 0.1]

    dataset_type = args.dataset_type

    experiment_name = args.exp_name

    mask_type = args.mask_type

    randomizer_mode = args.randomizer_mode

    randomization_prob = 0.75
    photo_bg_prob = 1.

    # sets the number of workers for dataloader, set to 0 for windows
    num_workers = args.num_workers

    dataset_path_train = args.traindata
    dataset_path_test = args.testdata
    bg_path = args.bg_path

    # original images have size (848x480), network input has size (848/input_scale,480/input_scale)
    input_scale = 2

    # use sigma = 2 and radius = 2 for 424x240, use sigma = 4 and radius = 4 for 848x480
    sigma = 2
    radius = 2

    gpu_device = args.gpu_device

    # test_set_selection = 'fraction'/'subset' if 'fraction' - all types of images are selected for training and
    # random fraction of are selected as a test set, if 'subset' - specify which types of images
    # are selected for train and test, spoecify in lists below
    test_set_selection = 'subset'


    # determine what parts of dataset to include
    def proc_args(inp):
        return list(inp.strip('[]').split(','))


    dataset_subset = ['Train']
    subject = proc_args(args.subject)
    camera = proc_args(args.camera)
    task = ['random']
    hand = proc_args(args.hand)
    object_type = [dataset_type]

    dataset_subset_test = ['Test']
    subject_test = proc_args(args.subject_test)
    camera_test = proc_args(args.camera_test)
    task_test = proc_args(args.task_test)
    hand_test = proc_args(args.hand_test)
    object_type_test = [dataset_type]

    attributes_train = [dataset_subset, subject, camera, task, hand, object_type, mask_type]
    attributes_test = [dataset_subset_test, subject_test, camera_test, task_test, hand_test, object_type_test,
                       mask_type]

    # initialize network
    net = dope_net(lr, gpu_device)  # switch dope_net for your own network

    # Create Imitrob Train dataset
    dataset = imitrob_dataset(dataset_path_train, bg_path, 'train', test_set_selection,
                              randomizer_mode, mask_type, True, True, test_examples_fraction_train,
                              attributes_train, attributes_test,
                              randomization_prob, photo_bg_prob,
                              input_scale, sigma, radius)

    # Create Imitrob Test dataset
    dataset_test = imitrob_dataset(dataset_path_test, bg_path, 'test', test_set_selection,
                                   randomizer_mode, mask_type, False, False, test_examples_fraction_test,
                                   attributes_train, attributes_test,
                                   0., 0.,
                                   input_scale, sigma, radius)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=True, num_workers=num_workers)

    # create log folder in results folder and write training info into log file
    cwd = os.getcwd()

    logs_dir = os.path.join(cwd, 'results')

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
    f.write('Training started at:' + (time.strftime("%Y_%m_%d___%H_%M_%S") + '\n'))
    f.write('========================================================' + '\n')

    train()
    if not args.skip_testing:
        test_model(net.load_model(os.path.join(logdir, 'checkpoint.pth.tar')), args=args)