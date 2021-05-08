# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2021/4/6 
# version： Python 3.7.8
# @File : draw_corr.py
# @Software: PyCharm
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import svm
import math
import matplotlib.pyplot as plt

data = pd.read_csv('file/adni1_bl/correlation_37.csv')

plt.rc('font', family='Times New Roman')
g = sns.jointplot(data['y'].tolist(), data['x'].tolist(), kind="reg",
                  xlim=(22, 30), ylim=(15, 30), color="mediumslateblue", height=7, marginal_kws=dict(bins=15, rug=False))

plt.ylabel('Predicted Value')
plt.xlabel('Ground Truth')
plt.text(22.2, 29.5, "BL", fontdict={'size': '14', 'color': 'k'})
plt.text(28.5, 15.5, "CC=0.577", fontdict={'size': '14', 'color': 'mediumslateblue'})
plt.show()
