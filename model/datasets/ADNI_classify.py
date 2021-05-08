import math
import os
import random
import numpy as np
from torch.utils.data import Dataset
import nibabel
from scipy import ndimage
import pandas as pd
from deepbrain import Extractor
from skimage import measure
import itertools
from PIL import Image, ImageEnhance
from skimage.util import random_noise
import nibabel as nib


class data_flow(Dataset):

    def __init__(self, image_path, train_sub_idx, train_labels, landmark_path,group,
                 patch_size, landmk_num, numofscales):

        self.image_path = image_path
        self.train_sub_idx = train_sub_idx
        self.train_labels = train_labels
        self.group = group
        self.landmk_num = landmk_num
        self.patch_size = patch_size
        self.numofscales = numofscales
        self.landmark_path = landmark_path

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, x, y, z])
        new_data = new_data.astype("float32")

        return new_data

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):

        paddingsize = 24

        # Load the landmarks
        # new_landmark_test = np.load(self.landmark_path + "landmarks_" + self.train_sub_idx[idx] + ".npy")
        new_landmark_test = np.load("/data1/qiaohezhe/MRI_MMSE/BL818_landmark/landmarks_002_S_0295_I118671.npy")
        new_landmark_test = new_landmark_test[:40, :].T
        landmarks = new_landmark_test.reshape(1, 3, 40)

        patch_size = self.patch_size
        numofscales = self.numofscales
        landmarks = np.round(landmarks) + paddingsize
        landmk_num = np.size(landmarks, 2)
        patches = np.zeros((landmk_num, numofscales, patch_size, patch_size, patch_size),
                           dtype='uint8')

        image_subject = nib.load(self.image_path + self.train_sub_idx[idx] + ".nii").get_data()
        image_subject = self.__scaler__(image_subject)
        image_subject = self.__resize_data__(image_subject)
        output = self.train_labels[idx]
        s_group = self.group[idx]
        if s_group == "CN":
            group = 1
        else:
            group = 0
        image_subject = np.lib.pad(image_subject, (
            (paddingsize, paddingsize), (paddingsize, paddingsize), (paddingsize, paddingsize)), 'constant',
                                   constant_values=0)

        for i_landmk in range(0, landmk_num):

            i_x = np.random.permutation(1)[0] + landmarks[0, 0, i_landmk]
            i_y = np.random.permutation(1)[0] + landmarks[0, 1, i_landmk]
            i_z = np.random.permutation(1)[0] + landmarks[0, 2, i_landmk]
            for i_scale in range(numofscales):
                patches[i_landmk, i_scale, 0:patch_size, 0:patch_size, 0:patch_size] = image_subject[int(
                    i_x - np.floor(patch_size * (i_scale + 1) / 2)):int(
                    i_x + int(np.ceil(patch_size * (i_scale + 1) / 2.0))):i_scale + 1,
                                                                                       int(i_y - np.floor(
                                                                                           patch_size * (
                                                                                                   i_scale + 1) / 2)):int(
                                                                                           i_y + int(
                                                                                               np.ceil(
                                                                                                   patch_size * (
                                                                                                           i_scale + 1) / 2.0))):i_scale + 1,
                                                                                       int(i_z - np.floor(
                                                                                           patch_size * (
                                                                                                   i_scale + 1) / 2)):int(
                                                                                           i_z + int(
                                                                                               np.ceil(
                                                                                                   patch_size * (
                                                                                                           i_scale + 1) / 2.0))):i_scale + 1]

        return patches, output

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __scaler__(self, image):
        img_f = image.flatten()
        # find the range of the pixel values
        i_range = img_f[np.argmax(img_f)] - img_f[np.argmin(img_f)]
        # clear the minus pixel values in images
        image = image - img_f[np.argmin(img_f)]
        img_normalized = np.float32(image / i_range)
        return img_normalized

    def __resize_data__(self, data):
        [depth, height, width] = data.shape
        scale = [256 * 1.0 / depth, 256 * 1.0 / height, 166 * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __crop_data__(self, data):
        # random center crop
        data = self.__random_center_crop__(data)

        return data

    def __training_data_process__(self, data, id):
        # crop data according net input size

        data = data.get_data()
        data = self.__drop_invalid_range__(data)
        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        # data = self.__itensity_normalize_one_volume__(data)

        data = self.__scaler__(data)

        return data

    def __drop_invalid_range__(self, volume, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)

        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)

        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]
