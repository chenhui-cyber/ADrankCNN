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
from sklearn.feature_extraction.image import extract_patches


class data_flow(Dataset):

    def __init__(self, image_path, train_sub_idx, train_labels, group):

        self.image_path = image_path
        self.train_sub_idx = train_sub_idx
        self.train_labels = train_labels
        self.group = group

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, x, y, z])
        new_data = new_data.astype("float32")

        return new_data

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        output = self.train_labels[idx]
        s_group = self.group[idx]
        if s_group == "CN":
            group = 1
        else:
            group = 0

        image_subject = nib.load(self.image_path  + self.train_sub_idx[idx] + ".nii")
        image_subject = self.__training_data_process__(image_subject, self.train_sub_idx[idx])
        patch_shape = (36, 36, 36)
        sub_patch_shape = (12, 12, 12)
        extraction_step = 10
        sub_extraction_step = 4
        landmk_num = 6 * 6 * 6
        # sub_landmk_num = 7 * 7 * 7
        patches = np.zeros((landmk_num, 1, patch_shape[0], patch_shape[1], patch_shape[2]),
                           dtype='uint8')
        # sub_patches = np.zeros((landmk_num,  sub_landmk_num, 1, sub_patch_shape[0], sub_patch_shape[1], sub_patch_shape[2]),
        #                    dtype='uint8')
        img_patches = extract_patches(image_subject, patch_shape, extraction_step)

        landmk_index = 0
        for i in range(img_patches.shape[0]):
            for j in range(img_patches.shape[1]):
                for k in range(img_patches.shape[2]):
                    patches[landmk_index, 0, ...] = img_patches[i, j, k, ...]

                    # sub_landmk_index = 0
                    # sub_img_patches = extract_patches(img_patches[i, j, k, ...], sub_patch_shape, sub_extraction_step)
                    # for sub_i in range(sub_img_patches.shape[0]):
                    #     for sub_j in range(sub_img_patches.shape[1]):
                    #         for sub_k in range(sub_img_patches.shape[2]):
                    #             sub_patches[landmk_index, sub_landmk_index, 0, ...] = sub_img_patches[i, j, k, ...]
                    #             sub_landmk_index += 1
                    landmk_index += 1

        return patches, output, group, np.expand_dims(image_subject, axis=0)

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
        scale = [90 * 1.0 / depth, 90 * 1.0 / height, 90 * 1.0 / width]
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __crop_data__(self, data):
        # random center crop
        data = self.__random_center_crop__(data)

        return data

    def __training_data_process__(self, data, id):
        # crop data according net input size
        print(id)
        data = data.get_data()
        data = self.__drop_invalid_range__(data)
        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        # data = self.__itensity_normalize_one_volume__(data)

        # data = self.__scaler__(data)

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
