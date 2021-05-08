# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2021/3/20 
# version： Python 3.7.8
# @File : voxel.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import nibabel as nib
from scipy import ndimage


def drop_invalid_range(volume, label=None):
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


def resize_data(data):
    [depth, height, width] = data.shape
    scale = [10 * 1.0 / depth, 10 * 1.0 / height, 10 * 1.0 / width]
    data = ndimage.interpolation.zoom(data, scale, order=0)
    return data


IMAGE1_PATH = "/data1/qiaohezhe/MRI_MMSE/BL818_GM/"
IMAGE2_PATH = "/data1/qiaohezhe/MRI_MMSE/BL776_GM/"

ADNI1_DATA = pd.read_csv("/data1/qiaohezhe/MRI_MMSE/time_data/ADNIMERGE_ADNI1_24_PROCESSED.csv")
ADNI2_DATA = pd.read_csv("/data1/qiaohezhe/MRI_MMSE/time_data/ADNIMERGE_ADNI2_24_PROCESSED.csv")

ADNI1_LBLS_BL = ADNI1_DATA['MMSE_BL'].tolist()
ADNI2_LBLS_BL = ADNI2_DATA['MMSE_BL'].tolist()

ADNI1_LBLS_06 = ADNI1_DATA['MMSE_06'].tolist()
ADNI2_LBLS_06 = ADNI2_DATA['MMSE_06'].tolist()

ADNI1_LBLS_12 = ADNI1_DATA['MMSE_12'].tolist()
ADNI2_LBLS_12 = ADNI2_DATA['MMSE_12'].tolist()

ADNI1_LBLS_24 = ADNI1_DATA['MMSE_24'].tolist()
ADNI2_LBLS_24 = ADNI2_DATA['MMSE_24'].tolist()

ADNI1_IDXS = ADNI1_DATA['SID'].tolist()
ADNI2_IDXS = ADNI2_DATA['SID'].tolist()
ADNI1_TPS = ADNI1_DATA['DX_bl'].tolist()
ADNI2_TPS = ADNI2_DATA['DX_bl'].tolist()

voxel_subject = []
for idx in range(len(ADNI2_LBLS_BL)):
    image_subject = nib.load(IMAGE2_PATH + 'c1' + ADNI2_IDXS[idx] + ".nii").get_data()
    image_subject = drop_invalid_range(image_subject)
    image_subject = resize_data(image_subject)
    voxel = [ADNI2_IDXS[idx], ADNI2_LBLS_BL[idx], ADNI2_LBLS_06[idx], ADNI2_LBLS_12[idx], ADNI2_LBLS_24[idx], ADNI2_TPS[idx]]
    for i in range(image_subject.shape[0]):
        for j in range(image_subject.shape[1]):
            for h in range(image_subject.shape[2]):
                voxel.append(image_subject[i, j, h])
    print(idx)
    voxel_subject.append(voxel)

data = pd.DataFrame(voxel_subject)
data.to_csv('/data1/qiaohezhe/wise_dnn/voxel/ADNIMERGE_ADNI2_VOXEL.csv', index=False, header=False)

