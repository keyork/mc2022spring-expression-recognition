#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :test.py
@说明        :执行单张样本测试
@时间        :2020/02/23 12:14:50
@作者        :钱彬
@版本        :1.0
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from resplutils.utils import *
from torch import nn
from resplutils.models import SRResNet,Generator
import time
from PIL import Image

# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 4      # 放大比例
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

source_path = '../data'
target_path = './'


def process_one_img(model, imgPath, save_path):

    # 加载图像
    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')

    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)

    # 转移数据至设备
    lr_img = lr_img.cuda()  # (1, 3, w, h ), imagenet-normed

    # 模型推理
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]   
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_img.save(save_path)

if __name__ == '__main__':
    
    
    # 预训练模型
    srgan_checkpoint = "./resplutils/results/checkpoint_srgan.pth"
    #srresnet_checkpoint = "./results/checkpoint_srresnet.pth"

    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(srgan_checkpoint)
    generator = Generator(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    generator = generator.cuda()
    generator.load_state_dict(checkpoint['generator'])

    generator.eval()
    model = generator

    for set_dir in ['train', 'val', 'test']:
        if not os.path.exists(set_dir):
            os.mkdir(set_dir)

    for dataset in os.listdir(source_path):
        dataset_path = os.path.join(source_path, dataset)
        for expr in os.listdir(dataset_path):
            print(expr)
            expr_path = os.path.join(dataset_path, expr)
            if not os.path.exists(os.path.join(target_path, dataset, expr)):
                os.mkdir(os.path.join(target_path, dataset, expr))
            for img in os.listdir(expr_path):
                source_img_path = os.path.join(expr_path, img)
                target_img_path = os.path.join(target_path, dataset, expr, img)
                process_one_img(model, source_img_path, target_img_path)