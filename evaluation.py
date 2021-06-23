import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from object_detector import find_objects
from cuboid_PNP_solver import CuboidPNPSolver
from error_metrics import rot_trans_err, ADD_error, calculate_AUC
import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from imitrob_dataset import imitrob_dataset
from trainer import proc_args

currenttime = time.strftime("%Y_%m_%d___%H_%M_%S")
timestamp_start = time.time()
logdir = ""
# NOTE: AUC_test_thresh is used to calculate acc after every epoch, AUC_test_thresh_2 is only for information purpuses
AUC_test_thresh = 0.02
AUC_test_thresh_2 = 0.1
AUC_test_thresh_range = [0., 0.1]
# original images have size (848x480), network input has size (848/input_scale,480/input_scale)
input_scale = 2

parser = argparse.ArgumentParser(description='Compute Accuracy')
parser.add_argument('--testdata', type=str, default='',
                    help='Path to the Test directory of Imitrob')
parser.add_argument('--bg_path', type=str, default="",
                    help='Path to the backgrounds folder')
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
parser.add_argument('--subject_test', type=str, default="",
                    help='List of subjects to be used for testing. All subjects: S1,S2,S3,S4')
parser.add_argument('--camera_test', type=str, default="",
                    help='List of cameras to be used for testing. All cameras: C1,C2')
parser.add_argument('--hand_test', type=str, default="",
                    help='List of hands to be used for testing. All hands: LH,RH')
parser.add_argument('--task_test', type=str, default="",
                    help='List of tasks to be used for testing. All tasks: clutter,round,sweep,press,frame,sparsewave,densewave')

def process_args(args=None):
    if args is None:
        args = parser.parse_args()
    num_workers = args.num_workers
    dataset_subset_test = ['Test']
    subject_test = proc_args(args.subject_test)
    camera_test = proc_args(args.camera_test)
    task_test = proc_args(args.task_test)
    hand_test = proc_args(args.hand_test)
    object_type_test = [args.dataset_type]
    mask_type = args.mask_type
    batch_size_test = args.batch_size_test
    attributes_test = [dataset_subset_test, subject_test, camera_test, task_test, hand_test, object_type_test, mask_type]
    dataset_path_test = args.testdata
    bg_path = args.bg_path
    test_set_selection = 'subset'
    randomizer_mode = args.randomizer_mode
    sigma = 2
    radius = 2
    test_examples_fraction_test = 1.
    dataset_test = imitrob_dataset(dataset_path_test, bg_path, 'test', test_set_selection,
                                   randomizer_mode, mask_type, False, False, test_examples_fraction_test,
                                   [0,0,0,0,0,0], attributes_test, 0., 0., input_scale, sigma, radius)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=True, num_workers=num_workers)
    return dataloader_test

def compute_loss(output_belief, output_affinities, target_belief, target_affinity):
    loss = None
    for l in output_belief:  # output each belief map layer.
        if loss is None:
            loss = ((l - target_belief) * (l - target_belief)).mean()
        else:
            loss_tmp = ((l - target_belief) * (l - target_belief)).mean()
            loss += loss_tmp

    # Affinities loss
    for l in output_affinities:  # output, each belief map layers.
        loss_tmp = ((l - target_affinity) * (l - target_affinity)).mean()
        loss += loss_tmp
    return loss

def test_model(model, test_images, test_affinities, test_beliefs, args):
    """
    Parameters:
    model: object with the trained model
    test_images: batch of images (float32), size: (test_batch_size,3,x,y)
    test_affinities: batch of affinity maps (float32), size: (test_batch_size,16,x/8,y/8)
    test_beliefs: batch of belief maps (float32), size: (test_batch_size,9,x/8,y/8)

    Returns:
    loss: scalar
    belief: output belief maps, size: size: (test_batch_size,9,x/8,y/8)
    affinity: output affinity maps, size: (test_batch_size,16,x/8,y/8)
    """
    if torch.cuda.is_available():
        test_images_v = Variable(test_images.cuda(device=args.gpu_device))
        test_beliefs_v = Variable(test_beliefs.cuda(device=args.gpu_device))
        test_affinities_v = Variable(test_affinities.cuda(device=args.gpu_device))
    else:
        test_images_v = Variable(test_images)
        test_beliefs_v = Variable(test_beliefs)
        test_affinities_v = Variable(test_affinities)

    with torch.no_grad():
        output_belief, output_affinity = model.forward(test_images_v)

    J = compute_loss(output_belief, output_affinity, test_beliefs_v, test_affinities_v)
    belief = output_belief[5].data.cpu().numpy()
    affinity = output_affinity[5].data.cpu().numpy()
    loss = J.data.cpu().numpy()
    return belief, affinity, loss


def test_batch_iterator(model, dataloader_test, args):
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

        belief_test, affinity_test, loss_test = test_model(model, test_images, test_affinities, test_beliefs, args)
        belief_test = belief_test.clip(min=0., max=1.)

        for j in range(len(belief_test)):
            cuboid2d, _ = find_objects(belief_test[j, :, :, :].astype(np.float64),
                                       affinity_test[j, :, :, :].astype(np.float64), 1, input_scale)

            RTT_matrix_gt_buffer.append(RTT_matrix_gt[j, :, :])
            bb3d_gt_buffer.append(bb3d_gt[j, :, :])

            info_buffer.append(
                [batch_label_info[0][j], batch_label_info[1][j], batch_label_info[2][j], batch_label_info[3][j]])
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
                vertices = np.concatenate((bb3d_defoult[j, :, :], centroid3d_defoult[j, :, :]))
                pnp_solver = CuboidPNPSolver(camera_intrinsic_matrix=internal_calibration_matrix[j, :, :],
                                             cuboid3d=vertices)
                location, quaternion, projected_points, RTT_matrix = pnp_solver.solve_pnp(cuboid2d)
                translation_err, rotation_err = rot_trans_err(RTT_matrix_gt[j, :, :], RTT_matrix)
                ADD_err = ADD_error(bb3d_gt[j, :, :], centroid3dgt[j, :, :], RTT_matrix, bb3d_defoult[j, :, :],
                                    centroid3d_defoult[j, :, :])
                translation_err_list_final.append(translation_err)
                rotation_err_list_final.append(rotation_err)
                ADD_err_list_final.append(ADD_err)
                projected_points_buffer.append(projected_points)
                cuboid2d_buffer.append(cuboid2d)
                RTT_matrix_buffer.append(RTT_matrix)
                location_buffer.append(location)
                quaternion_buffer.append(quaternion)
    err_metrics = {'translation_err_list_final': translation_err_list_final,
                   'rotation_err_list_final': rotation_err_list_final,
                   'ADD_err_list_final': ADD_err_list_final,
                   'object_finder_errors_test_final': object_finder_errors_test_final,
                   'cuboid_2d_buffer': cuboid2d_buffer,
                   'cuboid_3d_buffer': projected_points_buffer,
                   'RTT_matrix_buffer': RTT_matrix_buffer,
                   'location_buffer': location_buffer,
                   'quaternion_buffer': quaternion_buffer,
                   'RTT_matrix_gt_buffer': RTT_matrix_gt_buffer,
                   'bb3d_gt_buffer': bb3d_gt_buffer,
                   'info_buffer': info_buffer,
                   'file_buffer': file_buffer}
    return ADD_err_list_final, err_metrics

def generate_auc(ADD_err_list_final, err_metrics):
    AUC_acc_final = calculate_AUC(ADD_err_list_final, AUC_test_thresh)

    # here we generate AUC curve from: https://arxiv.org/pdf/1809.10790.pdf
    thresh = []
    acc_at_thresh = []

    for i in range(0, 1001):
        tr = AUC_test_thresh_range[0] + (AUC_test_thresh_range[1] / 1000) * i
        AUC = calculate_AUC(ADD_err_list_final, tr)

        thresh.append(tr)
        acc_at_thresh.append(AUC)

    plt.plot(thresh, acc_at_thresh)
    plt.ylim((0, 1))
    plt.xlabel('Average distance threshold in meters')
    plt.ylabel('ADD pass rate')

    # giving a title to my graph
    plt.title(currenttime + '_' + 'DOPE')
    plt.savefig(os.path.join(logdir, 'AUC_curve.png'), dpi=600)

    # function to show the plot
    plt.show()

    # save error metrics file to log folder in pickle file
    err_metrics['AUC_acc_final'] = AUC_acc_final
    err_metrics['ADD_err_list_final'] = ADD_err_list_final
    err_metrics['thresh'] = thresh
    err_metrics['acc_at_thresh'] = acc_at_thresh
    return err_metrics

def main(model, args=None):
    ''' input:
    model (obj): object with the learned weights. the eval script will run .forward on the model
    args (args obj): arguments if provided from another script. otherwise will use command-line arguments'''
    dataloader_test = proc_args(args)
    ADD_err_list_final, err_metrics = test_batch_iterator(model, dataloader_test, args)
    err_metrics = generate_auc(ADD_err_list_final, err_metrics)
    with open('err_metrics.pkl', 'wb') as f:
        pickle.dump(err_metrics, f)
        print("Results saved as {}".format('err_metrics.pkl'))


