
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from utils.getdlibfeat import *
from dataset.dlibfeatloader import DlibDataLoader
from models.xdlibmlp import XDlibMLP

from tensorboardX import SummaryWriter

def get_feat(config, expr_method):
    
    if os.path.exists(os.path.join(config.result_root, config.feat_path, config.train_data))\
        and os.path.exists(os.path.join(config.result_root, config.feat_path, config.test_data))\
        and os.path.exists(os.path.join(config.result_root, config.feat_path, config.val_data)):
        return 1

    predictor = dlib.shape_predictor(os.path.join(config.weight_root, config.predictor))
    convertor = dlib.face_recognition_model_v1(os.path.join(config.weight_root, config.face_descriptor))

    for dataset in config.subdir_list:
        print(dataset)
        data_path = os.path.join(config.data_path, dataset)
        # save_path = os.path.join(config.result_root, config.feat_path, dataset+'_sr.pth')
        save_path = os.path.join(config.result_root, config.feat_path, dataset+'.pth')
        expr_list = os.listdir(data_path)
        feat_lib = torch.tensor([0]).reshape(1, 1)
        for expr in expr_list:
            print(expr)
            label = config.expr_list.index(expr)
            expr_path = os.path.join(data_path, expr)
            img_list = os.listdir(expr_path)
            for img in img_list:
                if img[-3:] == 'jpg':
                    img_path = os.path.join(expr_path, img)
                    feature = img2feat(img_path, predictor, convertor)
                    img_tensor = torch.cat((torch.tensor([label]), torch.tensor(feature)), dim=0).reshape(1, 129)
                    if feat_lib.shape[1] == 1:
                        feat_lib = img_tensor
                    else:
                        feat_lib = torch.cat((feat_lib, img_tensor), dim=0)
        torch.save(feat_lib, save_path)


def train_dlib_feat(config, args_idx):
    
    args_list = [config.args1, config.args2, config.args3]
    train_args = args_list[args_idx]
    
    # load_data
    train_data = torch.load(os.path.join(config.result_root, config.feat_path, config.train_data))
    val_data = torch.load(os.path.join(config.result_root, config.feat_path, config.val_data))
    test_data = torch.load(os.path.join(config.result_root, config.feat_path, config.test_data))

    clfs = {
        'svm': svm.SVC(),
        'random_forest' : RandomForestClassifier(n_estimators=4096, max_depth=2048, n_jobs=-1, verbose=1)
        }
    
    for clf_key in clfs.keys():
        try:
            clf = clfs[clf_key]
            clf.fit(train_data[:, 1:], train_data[:, 0])
            result = torch.tensor([clf.predict(test_data[:, 1:])])
            acc = 1*(result == test_data[:, 0]).sum()/test_data.shape[0]
            print(clf_key)
            print('acc:{}'.format(acc))
        except:
            pass

    # BATCH_SIZE = config.batch_size

    # train_loader = DlibDataLoader(train_data, True, BATCH_SIZE)
    # val_loader = DlibDataLoader(val_data, False, BATCH_SIZE)
    # test_loader = DlibDataLoader(test_data, False, 1)

    # train_loader.load_data()
    # val_loader.load_data()
    # test_loader.load_data()

    # model = XDlibMLP()
    # model = model.cuda()

    # for m in model.modules():
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    # params_to_update = model.parameters()
    # print("Params to learn:")
    # params_to_update = []
    # for name,param in model.named_parameters():
    #     if param.requires_grad == True:
    #         params_to_update.append(param)
    #         print("\t",name)

    # loss_f = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(params_to_update, lr=train_args[0])
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_args[1], gamma=0.1, last_epoch=-1) # [50, 70, 90]

    # epoch_n = train_args[2]

    # exp_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime()) 
    # # model_writer = SummaryWriter(config.tensorboard_log + exp_time + '_dlibmlpsr')
    # model_writer = SummaryWriter(config.tensorboard_log + exp_time + '_dlibmlp')
    # for epoch in range(epoch_n):
    #     print('*'*10)
    #     print('Epoch {}/{}'.format(epoch + 1, epoch_n))
    #     print('-'*10)

    #     for phase in ['train', 'val']:
    #         if phase == 'train':
    #             model = model.train()
    #             data_loader = train_loader
    #         else:
    #             model = model.eval()
    #             data_loader = val_loader

    #         running_loss = 0.0
    #         running_corrects = 0

    #         for batch, data in enumerate(data_loader.dataloader, 1):
    #             X, y = data
    #             X, y = Variable(X), Variable(y)
    #             if torch.cuda.is_available():
    #                 X = X.cuda()
    #                 y = y.cuda()
    #             y_pred = model(X)
    #             _, pred = torch.max(y_pred.data, 1)
    #             optimizer.zero_grad()
    #             loss = loss_f(y_pred, y)
    #             if phase == 'train':
    #                 loss.backward()
    #                 optimizer.step()
    #             running_loss += loss.item()
    #             running_corrects += torch.sum(pred == y.data)
    #             if phase == 'train':
    #                 model_writer.add_scalar('training_loss', loss.item(), epoch*len(data_loader.dataloader)+batch)
    #                 model_writer.add_scalar('training_acc', torch.sum(pred == y.data)/BATCH_SIZE, epoch*len(data_loader.dataloader)+batch)
    #             if phase == 'val':
    #                 model_writer.add_scalar('val_loss', loss.item(), epoch*len(data_loader.dataloader)+batch)
    #                 model_writer.add_scalar('val_acc', torch.sum(pred == y.data)/BATCH_SIZE, epoch*len(data_loader.dataloader)+batch)
                
    #             # if batch%20 == 0 and phase == 'train':
    #             #     print('Batch {}, Training Loss:{:.4f}, Train Acc:{:.4f}'.\
    #             #         format(batch, running_loss/batch, float(100*running_corrects)/(BATCH_SIZE*batch)))
    #         epoch_loss = running_loss*BATCH_SIZE/len(data_loader.dataset)
    #         epoch_acc = float(100*running_corrects)/len(data_loader.dataset)
    #         print('{} Loss:{:.4f} Acc:{:.4f}%'.format(phase, epoch_loss, epoch_acc))

    #     scheduler.step()
    # model_writer.close()

    # # test
    # phase = 'test'
    # model = model.eval()
    # data_loader = test_loader

    # running_loss = 0.0
    # running_corrects = 0

    # for batch, data in enumerate(data_loader.dataloader, 1):
    #     X, y = data
    #     X, y = Variable(X), Variable(y)
    #     if torch.cuda.is_available():
    #         X = X.cuda()
    #         y = y.cuda()
    #     y_pred = model(X)
    #     _, pred = torch.max(y_pred.data, 1)
    #     loss = loss_f(y_pred, y)
    #     running_loss += loss.item()
    #     running_corrects += torch.sum(pred == y.data)
    # epoch_loss = running_loss*BATCH_SIZE/len(data_loader.dataset)
    # epoch_acc = float(100*running_corrects)/len(data_loader.dataset)
    # print('{} Loss:{:.4f} Acc:{:.4f}%'.format(phase, epoch_loss, epoch_acc))
    
    # SAVE_PATH = os.path.join(config.weight_root, 'Acc_'+str(epoch_acc)[:5].replace('.', '_')+config.model_basename)
    # torch.save(model.state_dict(), SAVE_PATH)