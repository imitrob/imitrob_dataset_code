#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 14:13:05 2019

@author: matusmacbookpro
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision

#adapted from: https://github.com/NVlabs/Deep_Object_Pose/blob/master/scripts/train.py


def create_stage(in_channels, out_channels, first=False):
    '''Create the neural network layers for a single stage.'''

    model = nn.Sequential()
    mid_channels = 128
    if first:
        padding = 1
        kernel = 3
        count = 6
        final_channels = 512
    else:
        padding = 3
        kernel = 7
        count = 10
        final_channels = mid_channels

    # First convolution
    model.add_module("0",
                     nn.Conv2d(
                         in_channels,
                         mid_channels,
                         kernel_size=kernel,
                         stride=1,
                         padding=padding)
                    )

    # Middle convolutions
    i = 1
    while i < count - 1:
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i),
                         nn.Conv2d(
                             mid_channels,
                             mid_channels,
                             kernel_size=kernel,
                             stride=1,
                             padding=padding))
        i += 1

    # Penultimate convolution
    model.add_module(str(i), nn.ReLU(inplace=True))
    i += 1
    model.add_module(str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
    i += 1

    # Last convolution
    model.add_module(str(i), nn.ReLU(inplace=True))
    i += 1
    model.add_module(str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
    i += 1

    return model


class Net(nn.Module):

    def __init__(self,pretrained=True,numBeliefMap=9,numAffinity=16,stop_at_stage=6):  # number of stages to process (if less than total number of stages):
        torch.manual_seed(123)

        super(Net, self).__init__()

        self.stop_at_stage = stop_at_stage

        vgg_full = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features

        self.vgg = nn.Sequential()

        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        # Add some layers
        i_layer = 23
        self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer+1), nn.ReLU(inplace=True))
        self.vgg.add_module(str(i_layer+2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer+3), nn.ReLU(inplace=True))


        # _2 are the belief map stages
        self.m1_2 = create_stage(128, numBeliefMap, True)
        self.m2_2 = create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m3_2 = create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m4_2 = create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m5_2 = create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m6_2 = create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)


        # _1 are the affinity map stages
        self.m1_1 = create_stage(128, numAffinity, True)
        self.m2_1 = create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m3_1 = create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m4_1 = create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m5_1 = create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m6_1 = create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)


    def forward(self, x):
        '''Runs inference on the neural network'''

        out1 = self.vgg(x)

        out1_2 = self.m1_2(out1)
        out1_1 = self.m1_1(out1)

        if self.stop_at_stage == 1:
            return [out1_2],\
                   [out1_1]

        out2 = torch.cat([out1_2, out1_1, out1], 1)
        out2_2 = self.m2_2(out2)
        out2_1 = self.m2_1(out2)

        if self.stop_at_stage == 2:
            return [out1_2, out2_2],\
                   [out1_1, out2_1]

        out3 = torch.cat([out2_2, out2_1, out1], 1)
        out3_2 = self.m3_2(out3)
        out3_1 = self.m3_1(out3)

        if self.stop_at_stage == 3:
            return [out1_2, out2_2, out3_2],\
                   [out1_1, out2_1, out3_1]

        out4 = torch.cat([out3_2, out3_1, out1], 1)
        out4_2 = self.m4_2(out4)
        out4_1 = self.m4_1(out4)

        if self.stop_at_stage == 4:
            return [out1_2, out2_2, out3_2, out4_2],\
                   [out1_1, out2_1, out3_1, out4_1]

        out5 = torch.cat([out4_2, out4_1, out1], 1)
        out5_2 = self.m5_2(out5)
        out5_1 = self.m5_1(out5)

        if self.stop_at_stage == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2],\
                   [out1_1, out2_1, out3_1, out4_1, out5_1]

        out6 = torch.cat([out5_2, out5_1, out1], 1)
        out6_2 = self.m6_2(out6)
        out6_1 = self.m6_1(out6)

        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2],\
               [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]


class dope_net():

    def __init__(self,learning_rate,gpu_device):
        self.cud = torch.cuda.is_available()

        self.gpu_device = gpu_device

        self.learning_rate = learning_rate

        self.net = Net()

        if self.cud:
            self.net.cuda(device=self.gpu_device)
            print(f"Using GPU device: {self.gpu_device}.")
        else:
            print("Using CPU.")

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)


    def compute_loss(self,output_belief,output_affinities,target_belief,target_affinity):

        loss = None

        for l in output_belief: #output, each belief map layers.
            if loss is None:
                loss = ((l - target_belief) * (l-target_belief)).mean()
            else:
                loss_tmp = ((l - target_belief) * (l-target_belief)).mean()
                loss += loss_tmp

        # Affinities loss
        for l in output_affinities: #output, each belief map layers.
            loss_tmp = ((l - target_affinity) * (l-target_affinity)).mean()
            loss += loss_tmp

        return loss


    def adjust_learning_rate(self,optimizer,lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self,train_images,train_affinities,train_beliefs):

#            INPUT:

#            train_images: batch of images (float32), size: (batch_size,3,x,y)
#            train_affinities: batch of affinity maps (float32), size: (batch_size,16,x/8,y/8)
#            train_beliefs: batch of belief maps (float32), size: (batch_size,9,x/8,y/8)

#            OUTPUTS:

#            loss: scalar


        if self.cud:
            train_images_v = Variable(train_images.cuda(device=self.gpu_device))
            train_affinities_v = Variable(train_affinities.cuda(device=self.gpu_device))
            train_beliefs_v = Variable(train_beliefs.cuda(device=self.gpu_device))
        else:
            train_images_v = Variable(train_images)
            train_affinities_v = Variable(train_affinities)
            train_beliefs_v = Variable(train_beliefs)


        self.optimizer.zero_grad()

        output_belief,output_affinity = self.net.forward(train_images_v)

        J = self.compute_loss(output_belief,output_affinity,train_beliefs_v,train_affinities_v)

        J.backward()

        self.optimizer.step()

        if self.cud:
            loss = J.data.cpu().numpy()
        else:
            loss = J.data.numpy()

        return loss


    def test(self,test_images,test_affinities,test_beliefs):

#            INPUT:

#            test_images: batch of images (float32), size: (test_batch_size,3,x,y)
#            test_affinities: batch of affinity maps (float32), size: (test_batch_size,16,x/8,y/8)
#            test_beliefs: batch of belief maps (float32), size: (test_batch_size,9,x/8,y/8)

#            OUTPUTS:

#            loss: scalar
#            belief: output belief maps, size: size: (test_batch_size,9,x/8,y/8)
#            affinity: output affinity maps, size: (test_batch_size,16,x/8,y/8)

        if self.cud:
            test_images_v = Variable(test_images.cuda(device=self.gpu_device))
            test_beliefs_v = Variable(test_beliefs.cuda(device=self.gpu_device))
            test_affinities_v = Variable(test_affinities.cuda(device=self.gpu_device))
        else:
            test_images_v = Variable(test_images)
            test_beliefs_v = Variable(test_beliefs)
            test_affinities_v = Variable(test_affinities)

        with torch.no_grad():
            output_belief,output_affinity = self.net.forward(test_images_v)

        J = self.compute_loss(output_belief,output_affinity,test_beliefs_v,test_affinities_v)

        if self.cud:
            belief = output_belief[5].data.cpu().numpy()
            affinity = output_affinity[5].data.cpu().numpy()
            loss = J.data.cpu().numpy()
        else:
            belief = output_belief[5].data.numpy()
            affinity = output_affinity[5].data.numpy()
            loss = J.data.numpy()

        return belief,affinity,loss

    def save_model(self,filename):
        torch.save(self.net.state_dict(),filename)

    def empty_cuda_cache(self):
        torch.cuda.empty_cache()

    def load_model(self,filename):
        if self.cud:
            self.net.load_state_dict(torch.load(filename,map_location = 'cuda:0'))
        else:
            self.net.load_state_dict(torch.load(filename,map_location='cpu'))