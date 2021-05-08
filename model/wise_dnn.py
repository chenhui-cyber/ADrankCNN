import torch
from tqdm import tqdm
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import math
import sys
import torch
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from datasets.ADNI_classify import data_flow

import time
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import torch.nn.functional as F
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [5]))
start = datetime.now()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class single_net(nn.Module):
    def __init__(self, f=4):
        super(single_net, self).__init__()
        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv1',
                               nn.Conv3d(in_channels=2, out_channels=4 * f, kernel_size=3, stride=2,
                                         padding=2,
                                         dilation=2))
        self.layer1.add_module('bn1', nn.BatchNorm3d(num_features=4 * f))
        self.layer1.add_module('relu1', nn.ReLU(inplace=True))
        self.layer1.add_module('max_pooling1', nn.MaxPool3d(kernel_size=2, stride=1))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv2',
                               nn.Conv3d(in_channels=4 * f, out_channels=8 * f, kernel_size=3, stride=2,
                                         padding=2,
                                         dilation=2))
        self.layer2.add_module('bn2', nn.BatchNorm3d(num_features=8 * f))
        self.layer2.add_module('relu2', nn.ReLU(inplace=True))
        self.layer2.add_module('max_pooling2', nn.MaxPool3d(kernel_size=2, stride=1))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv3',
                               nn.Conv3d(in_channels=8 * f, out_channels=16 * f, kernel_size=3, stride=2,
                                         padding=2,
                                         dilation=2))
        self.layer3.add_module('bn3', nn.BatchNorm3d(num_features=16 * f))
        self.layer3.add_module('relu3', nn.ReLU(inplace=True))
        self.layer3.add_module('max_pooling3', nn.MaxPool3d(kernel_size=2, stride=1))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(64, 24)
        self.fc2 = nn.Linear(24, 8)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class wiseNet(nn.Module):

    def __init__(self, f=8):
        super(wiseNet, self).__init__()

        self.fc3 = nn.Linear(40 * 8, 128)
        self.fc4 = nn.Linear(128, 8)
        self.fc5 = nn.Linear(8, 1)
        self.single_list = nn.ModuleList([single_net(f=4) for i in range(40)])
        self.single = single_net(f=4)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv1', nn.Conv3d(in_channels=1, out_channels=4 * f, kernel_size=3, stride=1, padding=0,
                                                  dilation=1))
        self.layer1.add_module('bn1', nn.BatchNorm3d(num_features=4 * f))
        self.layer1.add_module('relu1', nn.ReLU(inplace=True))
        self.layer1.add_module('max_pooling1', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv2',
                               nn.Conv3d(in_channels=4 * f, out_channels=16 * f, kernel_size=3, stride=1, padding=0,
                                         dilation=2))
        self.layer2.add_module('bn2', nn.BatchNorm3d(num_features=16 * f))
        self.layer2.add_module('relu2', nn.ReLU(inplace=True))
        self.layer2.add_module('max_pooling2', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv3',
                               nn.Conv3d(in_channels=16 * f, out_channels=32 * f, kernel_size=3, stride=1, padding=2,
                                         dilation=2))
        self.layer3.add_module('bn3', nn.BatchNorm3d(num_features=32 * f))
        self.layer3.add_module('relu3', nn.ReLU(inplace=True))
        self.layer3.add_module('max_pooling3', nn.MaxPool3d(kernel_size=3, stride=2))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv4',
                               nn.Conv3d(in_channels=32 * f, out_channels=64 * f, kernel_size=2, stride=1, padding=1,
                                         dilation=2))
        self.layer4.add_module('bn4', nn.BatchNorm3d(num_features=64 * f))
        self.layer4.add_module('relu4', nn.ReLU(inplace=True))
        self.layer4.add_module('max_pooling4', nn.MaxPool3d(kernel_size=5, stride=2))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential()
        self.fc.add_module('fc1', nn.Linear(256, 128))
        self.fc_sub = nn.Linear(686, 8)
        # self.sub_single = sub_single_net(f=4)

    def forward(self, x1):
        # x3 = self.layer1(x3)
        # x3 = self.layer2(x3)
        # x3 = self.layer3(x3)
        # x3 = self.layer4(x3)
        # x3 = self.avgpool(x3)
        # x3 = x3.view(x3.shape[0], -1)
        # x3 = self.fc(x3)
        print(x1.shape)
        branch = []

        for i_num in range(0,40):
            single_branch = self.single_list[i_num].forward(x1[:, i_num])
            branch.append(single_branch.unsqueeze(1))

        x1 = torch.cat(branch, dim=1)
        x1 = x1.view((x1.shape[0], -1))

        '''concate whole image'''
        # x1 = torch.cat((x1, x2), dim=1)
        merged = x1
        x1 = self.fc3(x1)
        x1 = self.fc4(x1)
        x1 = self.fc5(x1)
        x1 = x1.reshape(x1.shape[0])
        return x1, merged


if __name__ == "__main__":

    NUM_CHNS = 1
    FEATURE_DEPTH = [32, 64, 64, 128, 128, 64]
    NUM_REGION_FEATURES = 64
    NUM_SUBJECT_FEATURES = 64

    WITH_BN = True
    WITH_DROPOUT = True
    DROP_PROB = 0.5

    SHARING_WEIGHTS = False

    PATCH_SIZE = 24

    TRN_BATCH_SIZE = 24
    TST_BATCH_SIZE = 24

    NUM_EPOCHS = 100

    IMAGE1_PATH = "/data1/qiaohezhe/MRI_MMSE/BL818_processed/"
    IMAGE2_PATH = "/data1/qiaohezhe/MRI_MMSE/BL776_processed/"

    ADNI1_DATA = pd.read_csv("/data1/qiaohezhe/MRI_MMSE/time_data/ADNIMERGE_ADNI1_BL_PROCESSED.csv")
    ADNI2_DATA = pd.read_csv("/data1/qiaohezhe/MRI_MMSE/time_data/ADNIMERGE_ADNI2_BL_PROCESSED.csv")


    TRN_LBLS = ADNI1_DATA['MMSE'].tolist()
    VAL_LBLS = ADNI2_DATA['MMSE'].tolist()
    TRN_SUBJECT_IDXS = ADNI1_DATA['SID'].tolist()
    VAL_SUBJECT_IDXS = ADNI2_DATA['SID'].tolist()

    TRN_TPS = ADNI1_DATA['DX_bl'].tolist()
    VAL_TPS = ADNI2_DATA['DX_bl'].tolist()

    # TRN_SUBJECT_IDXS, VAL_SUBJECT_IDXS, TRN_LBLS, VAL_LBLS, TRN_TPS, VAL_TPS = \
    #     train_test_split(SUBJECT_IDXS, LABELS, TPS, test_size=0.3)

    print(len(TRN_SUBJECT_IDXS))
    print(len(VAL_SUBJECT_IDXS))
    TRN_STEPS = int(np.round(len(TRN_SUBJECT_IDXS) / TRN_BATCH_SIZE))
    TST_STEPS = int(np.round(len(VAL_SUBJECT_IDXS) / TST_BATCH_SIZE))

    # We set the patch sizes as 24*24*24 and 48*48*48 in our experiment.
    # The patches with the size of 48*48*48 will be downsampled into 24*24*24 to concatenate the small-scale patches
    patch_size = 48

    # In this code, we attached the pre-trained model which were trained with 40 landmarks
    landmk_num = 40
    # Since we used multi-scale patches, we need to set the scale into 2
    numofscales = 2
    # In the training stage, we used the these weights to normalize the range of outputs (from 0 to 1)
    weight = np.tile(np.array([10, 40, 60, 30]), 4)

    TRN_lANDMARK_PATH = "/data1/qiaohezhe/MRI_MMSE/BL818_landmark/"
    VAL_lANDMARK_PATH = "/data1/qiaohezhe/MRI_MMSE/BL776_landmark/"

    train_subject_num = len(TRN_SUBJECT_IDXS)
    val_subject_num = len(VAL_SUBJECT_IDXS)
    # Extract the multi-scale patches
    # Load the image
    # image = np.load('../Data/img.npy')
    train_flow = data_flow(IMAGE1_PATH, TRN_SUBJECT_IDXS, TRN_LBLS,
                                 TRN_lANDMARK_PATH, TRN_TPS, patch_size, train_subject_num, numofscales)
    test_flow = data_flow(IMAGE2_PATH, VAL_SUBJECT_IDXS, VAL_LBLS,
                              VAL_lANDMARK_PATH, VAL_TPS, patch_size, val_subject_num, numofscales)

    train_loader = DataLoader(dataset=train_flow, batch_size=24, shuffle=True)
    val_loader = DataLoader(dataset=test_flow, batch_size=24, shuffle=True)

    print("Train data load success")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = wiseNet(f=4)
    model = torch.nn.DataParallel(model)

    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-2)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    early_stopping = EarlyStopping(patience=80, verbose=True)
    model.to(device)
    result_list = []
    epochs = 80

    print("start training epoch {}".format(epochs))

    for epoch in range(epochs):
        print("Epoch{}:".format(epoch + 1))
        running_loss = 0
        running_loss_tripet = 0
        model.train()
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps, merged = model.forward(inputs.float())
            loss = F.mse_loss(logps, labels.float())
            print("step", i)

            print("train ground truth", labels)
            print("train predict", logps)

            loss.backward()
            print("loss_mmse", loss)
            optimizer.step()
            running_loss += loss.item()


        train_loss = running_loss / len(train_loader)
        print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch + 1, epochs, train_loss))

        running_loss = 0

        model.eval()
        with torch.no_grad():
            print("validation...")
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                logps, merged = model.forward(inputs.float())
                loss = F.mse_loss(logps, labels.float())

                print("train ground truth", labels)
                print("train predict", logps)
                print("loss_mmse", loss)

                running_loss += loss.item()

            val_loss = running_loss / len(val_loader)
            print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch + 1, epochs, val_loss))

            result_list.append([epoch, train_loss, val_loss])

            name = ['epoch', 'train_loss','val_loss']
            result = pd.DataFrame(columns=name, data=result_list)
            early_stopping(val_loss, model)

            result.to_csv("/data1/qiaohezhe/wise_dnn/log/land_mark.csv", mode='w',
                          index=False, header=False)
            if early_stopping.early_stop:
                # torch.save(model, "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/c3d_lstm/c3d_lstm.pth")
                print("Early stopping")
                break

            #         torch.save(model,
            #                    "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/c3d_lstm/c3d_lstm{}.pth".format(epoch))
            # torch.save(model, "/data1/qiaohezhe/MRI_Score/log/mci_nc/model/c3d_lstm/c3d_lstm.pth")

        stop = datetime.now()
        print("Running time: ", stop - start)
