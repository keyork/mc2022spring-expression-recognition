import os

from dataset import tiny_caltech35
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from model import base_model, feature_model
import torchvision.models as models
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
from torchvision.transforms import InterpolationMode
import seaborn as sns
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
exp_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime()) 

model = base_model(class_num=7)
model.load_state_dict(torch.load('./weights/model_baseline.pth'))
# feature_model = torch.nn.Sequential(*(list(model.children())[:-2]))
get_feature = feature_model(class_num=7)
get_feature.load_state_dict(torch.load('./weights/model_baseline.pth'))
# print(model)
# print(feature_model)
x = 0.5 * torch.ones(1,3,112,112)
print(model(x))
# print(feature_model(x).squeeze(dim=3).squeeze(dim=2))
print(get_feature(x))