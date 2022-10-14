
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.autograd import Variable

from config.cfgdlib import DlibConfig
from config.cfgcnn import CNNConfig

from utils.traindlibfeat import get_feat, train_dlib_feat
from utils.traincnn import train_cnn
from utils.inputbox import str2bool

def train(config, xconfig):

    expr_method = xconfig.method
    if expr_method == 'dlib_feat':
        # args_idx = 0, 1, 2
        train_dlib_feat(config, args_idx=xconfig.args_idx)
    elif expr_method == 'cnn':
        train_cnn(config, xconfig)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='dlib_feat')
    parser.add_argument('--network', type=str, default='ferckynet')
    parser.add_argument('--pretrained', type=str2bool, default=False)
    parser.add_argument('--args_idx', type=int, default=0)
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--step2', type=str2bool, default=False)
    parser.add_argument('--pre_model', type=str, default=None)
    xconfig = parser.parse_args()
    if xconfig.method == 'dlib_feat':
        config = DlibConfig()
        get_feat(config, xconfig)
        train(config, xconfig)
    elif xconfig.method == 'cnn':
        config = CNNConfig()
        train(config, xconfig)
