
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

BATCH_SIZE = 64
from dataset.xdataloader import XDataLoader

fer_loader = XDataLoader(root_path='./data', is_train=True, img_size=224, batch_size=BATCH_SIZE)
fer_loader.load_data()
# print(fer_loader.dataloader)
X_example, y_example = next(iter(fer_loader.dataloader['train']))
# print('X_example个数:{}'.format(len(X_example)))
# print(X_example)
# print('y_example个数:{}'.format(len(y_example)))
print(y_example)

from models.xresnet50 import XResNet50
from models.xvgg16 import XVGG16
from models.xalexnet import XAlexNet
from models.xmobinetv3 import XMobileNetV3

# model_loader = XResNet50(class_num=7, pre_trained=True)
# model_loader = XVGG16(class_num=7, pre_trained=True)
model_loader = XAlexNet(class_num=7, pre_trained=True)
# model_loader = XMobileNetV3(class_num=7, pre_trained=False)
model = model_loader.model
# print(model)
model = model.cuda()

import torch, torch.nn, torch.optim
import time
from torch.autograd import Variable
from tensorboardX import SummaryWriter
exp_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime()) 
# train_loss_writer = SummaryWriter('runs/' + exp_time + '/train_loss')
# train_acc_writer = SummaryWriter('runs/' + exp_time + '/train_acc')
# val_loss_writer = SummaryWriter('runs/' + exp_time + '/val_loss')
# val_acc_writer = SummaryWriter('runs/' + exp_time + '/val_acc')
model_writer = SummaryWriter('runs/' + exp_time + '/alexnet')

params_to_update = model.parameters()
print("Params to learn:")
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)

loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params_to_update, lr=3e-5, weight_decay=1)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 10, 20, 30, 40, 50, 55], gamma=0.33, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=10, verbose=True)


epoch_n = 2000
time_open = time.time()

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
                # train_loss_writer.add_scalar('training_loss', loss.item(), epoch*len(fer_loader.dataloader[phase])+batch)
                # train_acc_writer.add_scalar('training_acc', torch.sum(pred == y.data)/BATCH_SIZE, epoch*len(fer_loader.dataloader[phase])+batch)
                model_writer.add_scalar('training_loss', loss.item(), epoch*len(fer_loader.dataloader[phase])+batch)
                model_writer.add_scalar('training_acc', torch.sum(pred == y.data)/BATCH_SIZE, epoch*len(fer_loader.dataloader[phase])+batch)
            if phase == 'val':
                # val_loss_writer.add_scalar('val_loss', loss.item(), epoch*len(fer_loader.dataloader[phase])+batch)
                # val_acc_writer.add_scalar('val_acc', torch.sum(pred == y.data)/BATCH_SIZE, epoch*len(fer_loader.dataloader[phase])+batch)
                model_writer.add_scalar('val_loss', loss.item(), epoch*len(fer_loader.dataloader[phase])+batch)
                model_writer.add_scalar('val_acc', torch.sum(pred == y.data)/BATCH_SIZE, epoch*len(fer_loader.dataloader[phase])+batch)
            if batch%20 == 0 and phase == 'train':
                print('Batch {}, Training Loss:{:.4f}, Train Acc:{:.4f}'.\
                    format(batch, running_loss/batch, float(100*running_corrects)/(BATCH_SIZE*batch)))
        epoch_loss = running_loss*BATCH_SIZE/len(fer_loader.dataset[phase])
        epoch_acc = float(100*running_corrects)/len(fer_loader.dataset[phase])
        print('{} Loss:{:.4f} Acc:{:.4f}%'.format(phase, epoch_loss, epoch_acc))
    if phase == 'train':
        torch.save(model.state_dict(), 'model.pth')
    if phase == 'val':
        scheduler.step(epoch_loss)
time_end = time.time() - time_open
print(time_end)
# train_loss_writer.close()
# train_acc_writer.close()
# val_loss_writer.close()
# val_acc_writer.close()
model_writer.close()