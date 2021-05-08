# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2021/3/10 
# version： Python 3.7.8
# @File : mri_process.py
# @Software: PyCharm
# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/8/26
# version： Python 3.7.8
# @File : mri_process.py
# @Software: PyCharm

import sys
sys.path.append('/home/qiaohezhe/code/MRI_AD/util')
import os
# print(os.getcwd())
from deepbrain import Extractor
from image_util import *
from tqdm import tqdm
import nibabel as nb
import  numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [8]))


def skull_stripper(image):
    ext = Extractor()
    prob = ext.run(image)
    # set the threshold for extractor to extract the brain
    mask = prob < 0.7
    img_filtered = np.ma.masked_array(image, mask=mask)
    # fill the background to 0
    img_filtered = img_filtered.filled(0)

    return img_filtered


'''对每个受试者的最新MRI数据进行颅骨剥离'''
filePath = '/data1/qiaohezhe/MRI_MMSE/BL776/'
targetPath = '/data1/qiaohezhe/MRI_MMSE/BL776_processed/'
file_lsit = os.listdir(filePath)
for index, file in tqdm(enumerate(file_lsit), total=len(file_lsit)):
    print(file)
    img = nib.load(filePath + file)
    iskull = skull_stripper(np.array(img.get_data()))
    iskull = iskull.transpose((2, 1, 0))
    iskull = np.fliplr(iskull)
    iskull = np.flipud(iskull)
    new_image = nib.Nifti1Image(iskull, np.eye(4))
    # showNii(new_image)
    nb.save(new_image, targetPath + file)

'''对每个受试者的最新MRI数据进行颅骨剥离'''

# filePath = '/data1/qiaohezhe/miriad/mri/'
# savePath = '/data1/qiaohezhe/miriad/mri_processed/'
# file_lsit = os.listdir(filePath)
# print(filePath)
# for index, file in tqdm(enumerate(file_lsit), total=len(file_lsit)):
#         img = nib.load(filePath + file)
#         print(filePath + file)
#         try:
#             iskull = skull_stripper(np.array(img.get_data()))
#             new_image = nib.Nifti1Image(iskull, np.eye(4))
#             nb.save(new_image, savePath + file)
#         except Exception as e:
#             pass
#         continue

'''对MIMRIAD每个受试者的MRIj进行旋转配准'''
# filePath = '/data1/qiaohezhe/miriad/ad_nc/'
# savePath = '/data1/qiaohezhe/miriad/ad_nc_regsiter/'
# file_lsit = os.listdir(filePath)
# for index, file in tqdm(enumerate(file_lsit), total=len(file_lsit)):
#         img = nib.load(filePath + file)
#         img = img.get_data()
#         img = np.squeeze(img)
#         img = img.transpose((1, 2, 0))
#         # img = np.fliplr(img)
#         img = np.flipud(img)
#         img_data = nib.Nifti1Image(img, np.eye(4))
#         nib.save(img_data, savePath + file)


'''对adni2每个受试者的MRIj进行旋转配准'''
# filePath = '/data1/qiaohezhe/MRI_MMSE/BL776_processed/'
# savePath = '/data1/qiaohezhe/MRI_MMSE/BL776_processed/'
# file_lsit = os.listdir(filePath)
# for index, file in tqdm(enumerate(file_lsit), total=len(file_lsit)):
#         img = nib.load(filePath + file)
#         img = img.get_data()
#         img = np.squeeze(img)
#         img = img.transpose((2, 1, 0))
#         img = np.fliplr(img)
#         img = np.flipud(img)
#         img_data = nib.Nifti1Image(img, np.eye(4))
#         nib.save(img_data, savePath + file)
