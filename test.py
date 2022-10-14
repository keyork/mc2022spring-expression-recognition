
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import time
from tqdm import tqdm

from config.cfgdlib import DlibConfig
from config.cfgcnn import CNNConfig

from utils.traindlibfeat import get_feat, train_dlib_feat
from utils.traincnn import train_cnn
from utils.inputbox import str2bool
from utils.loadmodel import load_model
from utils.loadcfg import load_config
from utils.hidpt import HiddenPrints
from dataset.xdataloader import XDataLoader

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

import argparse

import mxnet as mx
from utils.mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector
import cv2


def test_dataset(model, config, xconfig, test_config):

    fer_test_loader = XDataLoader(root_path=config.data_path, is_train=False, img_size=test_config[0], batch_size=1)
    fer_test_loader.load_data()
    phase = 'test'
    model = model.eval()
    data_loader = fer_test_loader

    loss_f = nn.CrossEntropyLoss()
    BATCH_SIZE = test_config[3]

    running_loss = 0.0
    running_corrects = 0

    for batch, data in enumerate(tqdm(data_loader.dataloader[phase]), 1):
        X, y = data
        X, y = Variable(X), Variable(y)
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        y_pred = model(X)
        _, pred = torch.max(y_pred.data, 1)
        loss = loss_f(y_pred, y)
        running_loss += loss.item()
        running_corrects += torch.sum(pred == y.data)
    epoch_loss = running_loss*BATCH_SIZE/len(data_loader.dataset[phase])
    epoch_acc = float(100*running_corrects)/len(data_loader.dataset[phase])
    print('{} Loss:{:.4f} Acc:{:.4f}%'.format(phase, epoch_loss, epoch_acc))


def test_image(model, config, xconfig, test_config):

    base_img = Image.open(xconfig.img_path)
    raw_img = base_img.convert('RGB')
    transf = transforms.Compose([
                transforms.Resize([test_config[0], test_config[0]]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])
    img_tensor = transf(raw_img)
    model = model.eval()
    img_tensor = img_tensor.unsqueeze(0).cuda()
    y_pred = model(img_tensor)
    _, result = torch.max(y_pred.data, 1)
    print(config.class_name[result])


def real_time(model, config, xconfig, test_config):

    with HiddenPrints():
        detector = MtcnnDetector(model_folder='./utils/mxnet_mtcnn_face_detection/model', ctx=mx.cpu(), num_worker=4, accurate_landmark=False)
        camera = cv2.VideoCapture(0)
    while True:
        with HiddenPrints():
            grab, frame = camera.read()
            img = cv2.resize(frame, (720,480))

            t1 = time.time()
            results = detector.detect_face(img)
            print('time: ',time.time() - t1)

            if results is None:
                continue

            total_boxes = results[0]
            points = results[1]

            chips = detector.extract_image_chips(img, points, 144, 0.37)
        
            raw_img = Image.fromarray(chips[0], mode='RGB')
            transf = transforms.Compose([
                    transforms.Resize([test_config[0], test_config[0]]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ])
            img_tensor = transf(raw_img)
            model = model.eval()
            img_tensor = img_tensor.unsqueeze(0).cuda()
            y_pred = model(img_tensor)
            _, result = torch.max(y_pred.data, 1)
            text = config.class_name[result]
            
            font = cv2.FONT_HERSHEY_COMPLEX
            draw = img.copy()
            for b in total_boxes:
                cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (201, 161, 51), 2)
                cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[0]+90), int(b[1])+21), (201, 161, 51), -1)
                cv2.putText(draw, text, (int(b[0]), int(b[1])+13), font, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("detection result", draw)
            cv2.waitKey(30)
        
        print(config.class_name[result])


def test(xconfig):
    config = CNNConfig()
    test_config = load_config(config=config, network_name=xconfig.network)
    model_loader = load_model(network_name=xconfig.network, pretrain=None, class_num=config.class_num, layer_idx=test_config[8])
    model = model_loader.model
    model.load_state_dict(torch.load(os.path.join(config.weight_root, xconfig.weights)))
    model = model.cuda()
    if xconfig.method == 'dataset':
        test_dataset(model, config, xconfig, test_config)
    elif xconfig.method == 'single':
        test_image(model, config, xconfig, test_config)
    elif xconfig.method == 'realtime':
        real_time(model, config, xconfig, test_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='single')
    parser.add_argument('--network', type=str, default='vgg16')
    parser.add_argument('--weights', type=str, default='Acc_61_49_vgg16_sr.pth')
    parser.add_argument('--img_path', type=str, default='test.jpg')
    xconfig = parser.parse_args()
    test(xconfig)