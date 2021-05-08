# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2021/4/7 
# version： Python 3.7.8
# @File : caculate_corr.py
# @Software: PyCharm

import pandas as pd
import os
path = '/data1/qiaohezhe/wise_dnn/log/corr/adni2_bl/'

file_list = os.listdir(path)
result_data = []
for index in range(len(file_list)):
    data = pd.read_csv(path + 'correlation_' + str(index) + '.csv')
    ground = data['0']
    predict = data['1']
    result_data.append([index, ground.corr(predict)])

name = ['index', 'corr']
result = pd.DataFrame(columns=name, data=result_data)
result.to_csv('/data1/qiaohezhe/wise_dnn/log/corr/adni2_bl.csv', index = False)