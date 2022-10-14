
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import time

from dataset.xdataloader import XDataLoader
from utils.loadmodel import load_model
from utils.loadcfg import load_config

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

def train_cnn(config, xconfig):
    
    train_config = load_config(config=config, network_name=xconfig.network)
    BATCH_SIZE = train_config[3]

    # load model
    model_loader = load_model(network_name=xconfig.network, pretrain=xconfig.pretrained, class_num=config.class_num, layer_idx=train_config[8])
    model = model_loader.model
    if xconfig.step2:
        model.load_state_dict(torch.load(os.path.join(config.weight_root, xconfig.pre_model)))
    model = model.cuda()

    params_to_update = model.parameters()
    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)

    if xconfig.debug:
        network_writer = SummaryWriter('runs/our_network')
        dummy_input = torch.randn(1, 3, 192, 192).cuda()
        network_writer.add_graph(model, (dummy_input, ), True)
        network_writer.close()
        return 1
    else:
        pass

    # load data
    fer_loader = XDataLoader(root_path=config.data_path, is_train=True, img_size=train_config[0], batch_size=BATCH_SIZE)
    fer_loader.load_data()

    fer_test_loader = XDataLoader(root_path=config.data_path, is_train=False, img_size=train_config[0], batch_size=1)
    fer_test_loader.load_data()
    
    exp_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime()) 
    model_writer = SummaryWriter(config.tensorboard_log + exp_time + '_' + xconfig.network)
    # model_writer = SummaryWriter(config.tensorboard_log + exp_time + '_' + xconfig.network + 'sr')
    # model_writer = SummaryWriter(config.tensorboard_log + exp_time + '_' + xconfig.network + 'sr_step2')
    
    loss_f = nn.CrossEntropyLoss()
    if xconfig.step2:
        optimizer = optim.SGD(params_to_update, lr=0.001, weight_decay=0.0001, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1, last_epoch=-1)
        epoch_n = 60
    else:
        optimizer = optim.Adam(params_to_update, lr=train_config[1], weight_decay=train_config[6])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_config[2], gamma=train_config[7], last_epoch=-1)
        epoch_n = train_config[5]

    for epoch in range(epoch_n):
        print('*'*10)
        print('Epoch {}/{}'.format(epoch + 1, epoch_n))
        print('-'*10)

        for phase in ['train', 'val']:
            if phase == 'train':
                print("Training!")
                model = model.train()
            else:
                print('Validing!')
                model = model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch, data in enumerate(fer_loader.dataloader[phase], 1):
                X, y = data
                X, y = Variable(X), Variable(y)
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()
                y_pred = model(X)
                _, pred = torch.max(y_pred.data, 1)
                optimizer.zero_grad()
                loss = loss_f(y_pred, y)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                running_corrects += torch.sum(pred == y.data)
                if phase == 'train':
                    model_writer.add_scalar('training_loss', loss.item(), epoch*len(fer_loader.dataloader[phase])+batch)
                    model_writer.add_scalar('training_acc', torch.sum(pred == y.data)/BATCH_SIZE, epoch*len(fer_loader.dataloader[phase])+batch)
                if phase == 'val':
                    model_writer.add_scalar('val_loss', loss.item(), epoch*len(fer_loader.dataloader[phase])+batch)
                    model_writer.add_scalar('val_acc', torch.sum(pred == y.data)/BATCH_SIZE, epoch*len(fer_loader.dataloader[phase])+batch)
                if batch%50 == 0 and phase == 'train':
                    print('Batch {}, Training Loss:{:.4f}, Train Acc:{:.4f}'.\
                        format(batch, running_loss/batch, float(100*running_corrects)/(BATCH_SIZE*batch)))
            epoch_loss = running_loss*BATCH_SIZE/len(fer_loader.dataset[phase])
            epoch_acc = float(100*running_corrects)/len(fer_loader.dataset[phase])
            print('{} Loss:{:.4f} Acc:{:.4f}%'.format(phase, epoch_loss, epoch_acc))
        scheduler.step()
    model_writer.close()

    # test
    phase = 'test'
    model = model.eval()
    data_loader = fer_test_loader

    running_loss = 0.0
    running_corrects = 0

    for batch, data in enumerate(data_loader.dataloader[phase], 1):
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
    
    SAVE_PATH = os.path.join(config.weight_root, 'Acc_'+str(epoch_acc)[:5].replace('.', '_')+'_'+xconfig.network+'.pth')
    # SAVE_PATH = os.path.join(config.weight_root, 'Acc_'+str(epoch_acc)[:5].replace('.', '_')+'_'+xconfig.network+'_sr_step2.pth')
    torch.save(model.state_dict(), SAVE_PATH)