# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:03:31 2019

@author: Mat Tun
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from PIL import Image
import cv2
import os
from torchvision import transforms
from mask_green_background_bbox import rgba_from_rgb_green_bg,crop_hand_by_bbox


class BG_randomizer():

    def __init__(self,bg_path,max_width,max_hight,randomization_prob = 0.9,photo_bg_prob = 0.9):

#       randomization_prob determines the probability that green bg is swapped for different bg per image
#       photo_bg_prob determines the probability that green bg is swapped with random photo, othervise bg is swapped for random uniform color

        self.paths_images = []
        self.randomization_prob = randomization_prob
        self.photo_bg_prob = photo_bg_prob

        self.slash = os.sep
        self.source_folder = bg_path + self.slash

        self.max_width = max_width
        self.max_hight = max_hight

        if bg_path != '':

            for filename in glob.iglob(self.source_folder + '**/*.jpg', recursive=True):

                self.paths_images.append(os.path.abspath(filename))

#        debug code (find out how lowering number of bacground images impact accuracy)
#            does not seem to make much difference

#        no_background_fraction = 0.01
#
#        self.paths_images = random.sample(self.paths_images,int(len(self.paths_images)*no_background_fraction))
#        print(len(self.paths_images))


        # define torch transforms
        self.color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.rnd_color_jitter = transforms.RandomApply([self.color_jitter], p=0.8)
        self.rnd_gray = transforms.RandomGrayscale(p=0.2)

    def randomize_bg(self,image,bb2d,centroid2d,crop):

    #    SOURCE 1: https://stackoverflow.com/questions/52179821/python-3-i-am-trying-to-find-find-all-green-pixels-in-an-image-by-traversing-al
    #    SOURCE 2: https://github.com/kimmobrunfeldt/howto-everything/blob/master/remove-green.md

        # Open image and make RGB and HSV versions
        RGBim = image.convert('RGB')
        HSVim = RGBim.convert('HSV')

        # Make numpy versions
        RGBna = np.array(RGBim)
        HSVna = np.array(HSVim)

        if random.uniform(0,1) < self.randomization_prob:

            if random.uniform(0,1) < self.photo_bg_prob:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im = Image.open(bg_path)
                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

            else:

                bg_im = np.ones((RGBna.shape[0],RGBna.shape[1],3),dtype = np.uint8)

                bg_im[:,:,0] = bg_im[:,:,0]*random.randint(0,255)
                bg_im[:,:,1] = bg_im[:,:,1]*random.randint(0,255)
                bg_im[:,:,2] = bg_im[:,:,2]*random.randint(0,255)

        else:

            return RGBim,bb2d,centroid2d

        # Extract Hue
        H = HSVna[:,:,0]
        # Extract Saturation
        S = HSVna[:,:,1]
        # Extract Value
        V = HSVna[:,:,2]

        # Find all green pixels, i.e. where 100 < Hue < 140
        lo_h,hi_h = 100,140
        lo_s,hi_s = 80,255
        lo_v,hi_v = 70,255

        # Rescale S to 0-255, rather than 0-360 because we are using uint8
        lo_h = int((lo_h / 360) * 255)
        hi_h = int((hi_h / 360) * 255)

        green = np.where((H>=lo_h) & (H<=hi_h) & (S>=lo_s) & (S<=hi_s) & (V>=lo_v) & (V<=hi_v))

        RGBna[green] = bg_im[green]

        out = Image.fromarray(RGBna)

        if crop:

            out,bb2d,centroid2d = self.random_crop(out,bb2d,centroid2d)
            out,bb2d,centroid2d = self.flip_lr(out,bb2d,centroid2d)

        return out,bb2d,centroid2d


    def randomize_bg_bbox(self,image,pose,bb2d,centroid2d,crop):

        RGBimage = image.convert('RGB')

        RGBna = np.array(RGBimage)

        if random.uniform(0,1) < self.randomization_prob:

            if random.uniform(0,1) < self.photo_bg_prob:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im = Image.open(bg_path)
                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

            else:

                bg_im = np.ones((RGBna.shape[0],RGBna.shape[1],3),dtype = np.uint8)

                bg_im[:,:,0] = bg_im[:,:,0]*random.randint(0,255)
                bg_im[:,:,1] = bg_im[:,:,1]*random.randint(0,255)
                bg_im[:,:,2] = bg_im[:,:,2]*random.randint(0,255)

        else:

            return RGBimage,bb2d,centroid2d

        rgba, mask_green = rgba_from_rgb_green_bg(RGBna)

        rgba, mask_bbox = crop_hand_by_bbox(rgba, pose)

        mask = mask_green * mask_bbox

#        rgb = np.stack([rgba[:, :, 2],rgba[:, :, 1],rgba[:, :, 0]],axis=2).astype(np.uint8)
        rgb = np.stack([rgba[:, :, 0],rgba[:, :, 1],rgba[:, :, 2]],axis=2).astype(np.uint8)

        mask = np.stack([mask,mask,mask],axis=2)

        rgb = mask * rgb + (1. - mask) * bg_im

        rgb = rgb.astype(np.uint8)

        out = Image.fromarray(rgb)

        if crop:

            out,bb2d,centroid2d = self.random_crop(out,bb2d,centroid2d)
            out,bb2d,centroid2d = self.flip_lr(out,bb2d,centroid2d)


        return out,bb2d,centroid2d


    def randomize_bg_bbox_from_mask(self,image,mask,pose,bb2d,centroid2d,crop):

        RGBimage = image.convert('RGB')

        RGBna = np.array(RGBimage)

        if random.uniform(0,1) < self.randomization_prob:

            if random.uniform(0,1) < self.photo_bg_prob:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im = Image.open(bg_path)
                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

            else:

                bg_im = np.ones((RGBna.shape[0],RGBna.shape[1],3),dtype = np.uint8)

                bg_im[:,:,0] = bg_im[:,:,0]*random.randint(0,255)
                bg_im[:,:,1] = bg_im[:,:,1]*random.randint(0,255)
                bg_im[:,:,2] = bg_im[:,:,2]*random.randint(0,255)

        else:

            return RGBimage,bb2d,centroid2d

        mask = np.array(mask)
        # print(mask.shape)
        mask = mask[:,:,3]


        mask = np.stack([mask,mask,mask],axis=2)/255.


        out = mask * RGBna + (1. - mask) * bg_im

        out = out.astype(np.uint8)

        out = Image.fromarray(out)

        if crop:

            out,bb2d,centroid2d = self.random_crop(out,bb2d,centroid2d)
            out,bb2d,centroid2d = self.flip_lr(out,bb2d,centroid2d)


        return out,bb2d,centroid2d


    def randomize_bg_bbox_from_mask_empty_scene(self,image,image_empty_scene,mask,pose,bb2d,centroid2d,crop):

        RGBimage = image.convert('RGB')

        RGBna = np.array(RGBimage)

        if random.uniform(0,1) < self.randomization_prob:

            if random.uniform(0,1) < self.photo_bg_prob:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im = Image.open(bg_path)
                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

            else:

                # add bacground image of empty table
                bg_im = Image.open(image_empty_scene)
                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

        else:

            return RGBimage,bb2d,centroid2d

        mask = np.array(mask)
        # print(mask.shape)
        mask = mask[:,:,3]


        mask = np.stack([mask,mask,mask],axis=2)/255.


        out = mask * RGBna + (1. - mask) * bg_im

        out = out.astype(np.uint8)

        out = Image.fromarray(out)

        if crop:

            out,bb2d,centroid2d = self.random_crop(out,bb2d,centroid2d)
            out,bb2d,centroid2d = self.flip_lr(out,bb2d,centroid2d)


        return out,bb2d,centroid2d


    # this version is the same as previous version but randomization is NOT applied to bg image only to foreground
    def randomize_bg_bbox_from_mask_empty_scene_no_bg_aug(self,image,image_empty_scene,mask,pose,bb2d,centroid2d,crop):

        RGBimage = image.convert('RGB')

        RGBna = np.array(RGBimage)

        if random.uniform(0,1) < self.randomization_prob:

            if random.uniform(0,1) < self.photo_bg_prob:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im = Image.open(bg_path)
                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

            else:

                # add bacground image of empty table
                bg_im = Image.open(image_empty_scene)
                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

        else:

            return RGBimage,bb2d,centroid2d

        mask = np.array(mask)
        # print(mask.shape)
        mask = mask[:,:,3]


        mask = np.stack([mask,mask,mask],axis=2)

        if crop:

            RGBna = Image.fromarray(RGBna)
            mask = Image.fromarray(mask)

            RGBna,mask,bb2d,centroid2d = self.random_crop_w_mask(RGBna,mask,bb2d,centroid2d)
            RGBna,mask,bb2d,centroid2d = self.flip_lr_w_mask(RGBna,mask,bb2d,centroid2d)

            RGBna = np.array(RGBna)
            mask = np.array(mask)/255.


        out = mask * RGBna + (1. - mask) * bg_im

        out = out.astype(np.uint8)

        out = Image.fromarray(out)


        return out,bb2d,centroid2d


    def randomize_bg_bbox_from_mask_50_50_blend(self,image,image_empty_scene,mask,pose,bb2d,centroid2d,crop):

        RGBimage = image.convert('RGB')

        RGBna = np.array(RGBimage)

        if random.uniform(0,1) < self.randomization_prob:

            if random.uniform(0,1) < self.photo_bg_prob:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im = Image.open(bg_path)
                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

            else:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im_random = Image.open(bg_path)
                bg_im_random = bg_im_random.convert('RGB')
                bg_im_random = bg_im_random.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im_random = np.array(bg_im_random).astype(np.uint8)

                bg_im_empty_bg = Image.open(image_empty_scene)
                bg_im_empty_bg = bg_im_empty_bg.convert('RGB')
                bg_im_empty_bg = bg_im_empty_bg.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im_empty_bg = np.array(bg_im_empty_bg).astype(np.uint8)

                bg_im_random = Image.fromarray(bg_im_random)
                bg_im_empty_bg = Image.fromarray(bg_im_empty_bg)

                bg_im = Image.blend(bg_im_empty_bg,bg_im_random,0.5)
                bg_im = np.array(bg_im)

        else:

            return RGBimage,bb2d,centroid2d

        mask = np.array(mask)
        # print(mask.shape)
        mask = mask[:,:,3]


        mask = np.stack([mask,mask,mask],axis=2)/255.


        out = mask * RGBna + (1. - mask) * bg_im

        out = out.astype(np.uint8)

        out = Image.fromarray(out)

        if crop:

            out,bb2d,centroid2d = self.random_crop(out,bb2d,centroid2d)
            out,bb2d,centroid2d = self.flip_lr(out,bb2d,centroid2d)


        return out,bb2d,centroid2d


    def randomize_bg_bbox_from_mask_50_50_random_blend(self,image,image_empty_scene,mask,pose,bb2d,centroid2d,crop):

        RGBimage = image.convert('RGB')

        RGBna = np.array(RGBimage)

        if random.uniform(0,1) < self.randomization_prob:

            if random.uniform(0,1) < self.photo_bg_prob:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im = Image.open(bg_path)
                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

            else:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im_random = Image.open(bg_path)
                bg_im_random = bg_im_random.convert('RGB')
                bg_im_random = bg_im_random.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im_random = np.array(bg_im_random).astype(np.uint8)

                bg_im_empty_bg = Image.open(image_empty_scene)
                bg_im_empty_bg = bg_im_empty_bg.convert('RGB')
                bg_im_empty_bg = bg_im_empty_bg.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im_empty_bg = np.array(bg_im_empty_bg).astype(np.uint8)

                bg_im_random = Image.fromarray(bg_im_random)
                bg_im_empty_bg = Image.fromarray(bg_im_empty_bg)

                rand_blend_ratio = round(random.uniform(0,1),2)

                bg_im = Image.blend(bg_im_empty_bg,bg_im_random,rand_blend_ratio)
                bg_im = np.array(bg_im)

        else:

            return RGBimage,bb2d,centroid2d

        mask = np.array(mask)
        # print(mask.shape)
        mask = mask[:,:,3]


        mask = np.stack([mask,mask,mask],axis=2)/255.


        out = mask * RGBna + (1. - mask) * bg_im

        out = out.astype(np.uint8)

        out = Image.fromarray(out)

        if crop:

            out,bb2d,centroid2d = self.random_crop(out,bb2d,centroid2d)
            out,bb2d,centroid2d = self.flip_lr(out,bb2d,centroid2d)


        return out,bb2d,centroid2d


    def randomize_crop(self,image,bb2d,centroid2d):

        if random.uniform(0,1) < self.randomization_prob:

            image,bb2d,centroid2d = self.random_crop(image,bb2d,centroid2d)
            image,bb2d,centroid2d = self.flip_lr(image,bb2d,centroid2d)

            return image,bb2d,centroid2d

        else:

            return image,bb2d,centroid2d


    def randomize_bg_overlay(self,image,pose,bb2d,centroid2d,crop):

        RGBimage = image.convert('RGB')
#        RGBimage = image

        RGBna = np.array(RGBimage)

        if random.uniform(0,1) < self.randomization_prob:

            if random.uniform(0,1) < self.photo_bg_prob:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im = Image.open(bg_path)
                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

            else:

                bg_im = np.ones((RGBna.shape[0],RGBna.shape[1],3),dtype = np.uint8)

                bg_im[:,:,0] = bg_im[:,:,0]*random.randint(0,255)
                bg_im[:,:,1] = bg_im[:,:,1]*random.randint(0,255)
                bg_im[:,:,2] = bg_im[:,:,2]*random.randint(0,255)

        else:

            return RGBimage,bb2d,centroid2d

        rgba, mask_green = rgba_from_rgb_green_bg(RGBna)

        rgba, mask_bbox = crop_hand_by_bbox(rgba, pose)

        mask = mask_green * mask_bbox

#        rgb = np.stack([rgba[:, :, 2],rgba[:, :, 1],rgba[:, :, 0]],axis=2).astype(np.uint8)
        rgb = np.stack([rgba[:, :, 0],rgba[:, :, 1],rgba[:, :, 2]],axis=2).astype(np.uint8)

        mask = np.stack([mask,mask,mask],axis=2)


#        experimental: try superimposing background and image onto themselves

        background = (1. - mask) * bg_im

#        background = mask * rgb + (1. - mask) * bg_im

        rgb = rgb.astype(np.uint8)
        background = background.astype(np.uint8)

        rgb = Image.fromarray(rgb)
        background = Image.fromarray(background)

        out = Image.blend(background,rgb,0.5)


        if crop:

            out,bb2d,centroid2d = self.random_crop(out,bb2d,centroid2d)
            out,bb2d,centroid2d = self.flip_lr(out,bb2d,centroid2d)

        return out,bb2d,centroid2d


    def randomize_bg_overlay_from_mask(self,image,mask,pose,bb2d,centroid2d,crop,flip):

        RGBimage = image.convert('RGB')
#        RGBimage = image

        RGBna = np.array(RGBimage)

        if random.uniform(0,1) < self.randomization_prob:

            if random.uniform(0,1) < self.photo_bg_prob:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im = Image.open(bg_path)

                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

            else:

                bg_im = np.ones((RGBna.shape[0],RGBna.shape[1],3),dtype = np.uint8)

                bg_im[:,:,0] = bg_im[:,:,0]*random.randint(0,255)
                bg_im[:,:,1] = bg_im[:,:,1]*random.randint(0,255)
                bg_im[:,:,2] = bg_im[:,:,2]*random.randint(0,255)

        else:

            return RGBimage,bb2d,centroid2d

        mask = np.array(mask)
        mask = mask[:,:,3]

#        rgb = np.stack([rgba[:, :, 0],rgba[:, :, 1],rgba[:, :, 2]],axis=2).astype(np.uint8)
#        rgb = RGBna.astype(np.uint8)

        mask = np.stack([mask,mask,mask],axis=2)/255.


        background = mask * RGBna + (1. - mask) * bg_im

        RGBna = RGBna.astype(np.uint8)
        background = background.astype(np.uint8)

        RGBna = Image.fromarray(RGBna)
        background = Image.fromarray(background)

        out = Image.blend(background,RGBna,0.5)


        if crop:

            out,bb2d,centroid2d = self.random_crop(out,bb2d,centroid2d)

        if flip:

            out,bb2d,centroid2d = self.flip_lr(out,bb2d,centroid2d)

        # experimental feature
        # out = self.color_distort(out)

        return out,bb2d,centroid2d


    def randomize_bg_overlay_from_mask_noise_bg(self,image,mask,pose,bb2d,centroid2d,crop,flip):

        RGBimage = image.convert('RGB')
#        RGBimage = image

        RGBna = np.array(RGBimage)

        if random.uniform(0,1) < self.randomization_prob:

            if random.uniform(0,1) < self.photo_bg_prob:

                bg_im = np.random.random((RGBna.shape[0],RGBna.shape[1],3))
                bg_im = (bg_im*255).astype(np.uint8)

            else:

                bg_im = np.ones((RGBna.shape[0],RGBna.shape[1],3),dtype = np.uint8)

                bg_im[:,:,0] = bg_im[:,:,0]*random.randint(0,255)
                bg_im[:,:,1] = bg_im[:,:,1]*random.randint(0,255)
                bg_im[:,:,2] = bg_im[:,:,2]*random.randint(0,255)

        else:

            return RGBimage,bb2d,centroid2d

        mask = np.array(mask)
        mask = mask[:,:,3]

#        rgb = np.stack([rgba[:, :, 0],rgba[:, :, 1],rgba[:, :, 2]],axis=2).astype(np.uint8)
#        rgb = RGBna.astype(np.uint8)

        mask = np.stack([mask,mask,mask],axis=2)/255.


        background = mask * RGBna + (1. - mask) * bg_im

        RGBna = RGBna.astype(np.uint8)
        background = background.astype(np.uint8)

        RGBna = Image.fromarray(RGBna)
        background = Image.fromarray(background)

        out = Image.blend(background,RGBna,0.5)


        if crop:

            out,bb2d,centroid2d = self.random_crop(out,bb2d,centroid2d)

        if flip:

            out,bb2d,centroid2d = self.flip_lr(out,bb2d,centroid2d)

        return out,bb2d,centroid2d


    def randomize_bg_overlay_from_mask_empty_scene(self,image,image_empty_scene,mask,pose,bb2d,centroid2d,crop):

        RGBimage = image.convert('RGB')
#        RGBimage = image

        RGBna = np.array(RGBimage)

        if random.uniform(0,1) < self.randomization_prob:

            if random.uniform(0,1) < self.photo_bg_prob:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im = Image.open(bg_path)
                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

            else:

                # add bacground image of empty table
                bg_im = Image.open(image_empty_scene)
                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

        else:

            return RGBimage,bb2d,centroid2d

        mask = np.array(mask)
        mask = mask[:,:,3]

#        rgb = np.stack([rgba[:, :, 0],rgba[:, :, 1],rgba[:, :, 2]],axis=2).astype(np.uint8)
#        rgb = RGBna.astype(np.uint8)

        mask = np.stack([mask,mask,mask],axis=2)/255.


        background = mask * RGBna + (1. - mask) * bg_im

        RGBna = RGBna.astype(np.uint8)
        background = background.astype(np.uint8)

        RGBna = Image.fromarray(RGBna)
        background = Image.fromarray(background)

        out = Image.blend(background,RGBna,0.5)


        if crop:

            out,bb2d,centroid2d = self.random_crop(out,bb2d,centroid2d)
            out,bb2d,centroid2d = self.flip_lr(out,bb2d,centroid2d)

        return out,bb2d,centroid2d


    def randomize_bg_overlay_from_mask_50_50_blend(self,image,image_empty_scene,mask,pose,bb2d,centroid2d,crop):

        RGBimage = image.convert('RGB')
#        RGBimage = image

        RGBna = np.array(RGBimage)

        if random.uniform(0,1) < self.randomization_prob:

            if random.uniform(0,1) < self.photo_bg_prob:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im = Image.open(bg_path)
                bg_im = bg_im.convert('RGB')
                bg_im = bg_im.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im = np.array(bg_im)

            else:

                bg_path = random.sample(self.paths_images,1)[0]
                bg_im_random = Image.open(bg_path)
                bg_im_random = bg_im_random.convert('RGB')
                bg_im_random = bg_im_random.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im_random = np.array(bg_im_random).astype(np.uint8)

                bg_im_empty_bg = Image.open(image_empty_scene)
                bg_im_empty_bg = bg_im_empty_bg.convert('RGB')
                bg_im_empty_bg = bg_im_empty_bg.resize((RGBna.shape[1],RGBna.shape[0]),resample = Image.BILINEAR)
                bg_im_empty_bg = np.array(bg_im_empty_bg).astype(np.uint8)

                bg_im_random = Image.fromarray(bg_im_random)
                bg_im_empty_bg = Image.fromarray(bg_im_empty_bg)

                bg_im = Image.blend(bg_im_empty_bg,bg_im_random,0.5)
                bg_im = np.array(bg_im)

        else:

            return RGBimage,bb2d,centroid2d

        mask = np.array(mask)
        mask = mask[:,:,3]

#        rgb = np.stack([rgba[:, :, 0],rgba[:, :, 1],rgba[:, :, 2]],axis=2).astype(np.uint8)
#        rgb = RGBna.astype(np.uint8)

        mask = np.stack([mask,mask,mask],axis=2)/255.


        background = mask * RGBna + (1. - mask) * bg_im

        RGBna = RGBna.astype(np.uint8)
        background = background.astype(np.uint8)

        RGBna = Image.fromarray(RGBna)
        background = Image.fromarray(background)

        out = Image.blend(background,RGBna,0.5)


        if crop:

            out,bb2d,centroid2d = self.random_crop(out,bb2d,centroid2d)
            out,bb2d,centroid2d = self.flip_lr(out,bb2d,centroid2d)

        return out,bb2d,centroid2d



    def random_crop(self,image,bb2d,centroid2d):

        image_points = np.concatenate((bb2d,centroid2d),axis=0)

        border_dist_cutoff = 10
        max_iters = 20

        if (max(image_points[:,0])>((self.max_width-1)-border_dist_cutoff)) or (max(image_points[:,1])>((self.max_hight-1)-border_dist_cutoff)):

            return image,image_points[0:8,:],image_points[np.newaxis,8,:]

        p1 = max(image_points[:,0])
        p2 = min(image_points[:,1])
        p3 = min(image_points[:,0])
        p4 = max(image_points[:,1])

        success = 0

        asp_rat = self.max_width/self.max_hight

        for i in range(max_iters):

            C1_x = random.uniform(0,p3)
            C1_y = random.uniform(0,p2)

            y_limit = (self.max_hight-1)-C1_y
            x_limit = asp_rat*y_limit

            x_limit_y = asp_rat*(p4 - C1_y) + C1_x

            if (x_limit_y < x_limit) and (x_limit_y > p1):

                C2_x = random.uniform(x_limit_y,x_limit)

                C2_y = C1_y + (C2_x-C1_x)/asp_rat

                success = 1

                break

        if success == 0:

            return image,image_points[0:8,:],image_points[np.newaxis,8,:]

        image = image.crop((C1_x, C1_y, C2_x, C2_y))

        w,h = image.size

        image = image.resize((self.max_width,self.max_hight), Image.ANTIALIAS)

    #    RECALCULATE POINT LOCATIONS

    #    1. from 640*480 coordinate system to crop coordinate system

        new_image_points = np.zeros(image_points.shape,np.float32)

        new_image_points[:,0] = (image_points[:,0] - C1_x)
        new_image_points[:,1] = (image_points[:,1] - C1_y)

    #    2. from crop coordinate system to 640*480 coordinate system

        new_image_points[:,0] = (new_image_points[:,0]/w)*self.max_width
        new_image_points[:,1] = (new_image_points[:,1]/h)*self.max_hight


        return image,new_image_points[0:8,:],new_image_points[np.newaxis,8,:]


    def flip_lr(self,image,bb2d,centroid2d):

        image_points = np.concatenate((bb2d,centroid2d),axis=0)

        if random.uniform(0,1) > 0.5:

            w,h = image.size

            image = image.transpose(Image.FLIP_LEFT_RIGHT)

            image_points[:,0] = (w/2) - image_points[:,0] + (w/2)


        return image,image_points[0:8,:],image_points[np.newaxis,8,:]


    def random_crop_w_mask(self,image,mask,bb2d,centroid2d):

        image_points = np.concatenate((bb2d,centroid2d),axis=0)

        border_dist_cutoff = 10
        max_iters = 20

        if (max(image_points[:,0])>((self.max_width-1)-border_dist_cutoff)) or (max(image_points[:,1])>((self.max_hight-1)-border_dist_cutoff)):

            return image,mask,image_points[0:8,:],image_points[np.newaxis,8,:]

        p1 = max(image_points[:,0])
        p2 = min(image_points[:,1])
        p3 = min(image_points[:,0])
        p4 = max(image_points[:,1])

        success = 0

        asp_rat = self.max_width/self.max_hight

        for i in range(max_iters):

            C1_x = random.uniform(0,p3)
            C1_y = random.uniform(0,p2)

            y_limit = (self.max_hight-1)-C1_y
            x_limit = asp_rat*y_limit

            x_limit_y = asp_rat*(p4 - C1_y) + C1_x

            if (x_limit_y < x_limit) and (x_limit_y > p1):

                C2_x = random.uniform(x_limit_y,x_limit)

                C2_y = C1_y + (C2_x-C1_x)/asp_rat

                success = 1

                break

        if success == 0:

            return image,mask,image_points[0:8,:],image_points[np.newaxis,8,:]

        image = image.crop((C1_x, C1_y, C2_x, C2_y))
        mask = mask.crop((C1_x, C1_y, C2_x, C2_y))

        w,h = image.size

        image = image.resize((self.max_width,self.max_hight), Image.ANTIALIAS)
        mask = mask.resize((self.max_width,self.max_hight), Image.ANTIALIAS)

    #    RECALCULATE POINT LOCATIONS

    #    1. from 640*480 coordinate system to crop coordinate system

        new_image_points = np.zeros(image_points.shape,np.float32)

        new_image_points[:,0] = (image_points[:,0] - C1_x)
        new_image_points[:,1] = (image_points[:,1] - C1_y)

    #    2. from crop coordinate system to 640*480 coordinate system

        new_image_points[:,0] = (new_image_points[:,0]/w)*self.max_width
        new_image_points[:,1] = (new_image_points[:,1]/h)*self.max_hight


        return image,mask,new_image_points[0:8,:],new_image_points[np.newaxis,8,:]


    def flip_lr_w_mask(self,image,mask,bb2d,centroid2d):

        image_points = np.concatenate((bb2d,centroid2d),axis=0)

        if random.uniform(0,1) > 0.5:

            w,h = image.size

            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            image_points[:,0] = (w/2) - image_points[:,0] + (w/2)


        return image,mask,image_points[0:8,:],image_points[np.newaxis,8,:]


    def color_distort(self,image):

        image = self.color_jitter(image)
        image = self.rnd_gray(image)

        return image


#TEST

#slash = os.sep
#
#source = ''
#
#randomizer = BG_randomizer(''+ slash,1.)
#ep_sample_dir = ''
#
#paths_images = []
#
#for filename in glob.iglob(source + slash + '**/*.png', recursive=True):
#
#    if 'Depth' not in filename:
#
#        paths_images.append(filename)
#
#
#for i in range(50):
#
#    img_path = random.sample(paths_images,1)[0]
#
#    img = Image.open(img_path)
#
#    out_im = randomizer.randomize_bg_bbox(img)
#    plt.imshow(out_im)
#    plt.show()
#    cv2.imwrite(os.path.join(ep_sample_dir,'sample_' + str(i) + '.jpg'),cv2.cvtColor(np.array(out_im), cv2.COLOR_RGB2BGR))





