import os

from dataset import tiny_caltech35
import torchvision.transforms as transforms
import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
import argparse
from model import base_model, feature_model
import torchvision.models as models
from tensorboardX import SummaryWriter
# from sklearn.manifold import TSNE
from tsnecuda import TSNE
from sklearn.decomposition import PCA
from torchvision.transforms import InterpolationMode
import seaborn as sns
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
exp_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime()) 

network_writer = SummaryWriter('runs/' + exp_time + '/network')
train_loss_writer = SummaryWriter('runs/' + exp_time + '/train_loss')
train_acc_writer = SummaryWriter('runs/' + exp_time + '/train_acc')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(config):
    transform_train = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = tiny_caltech35(transform=transform_train, used_data=['train'])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)

    val_dataset = tiny_caltech35(transform=transform_test, used_data=['val'])
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)

    test_dataset = tiny_caltech35(transform=transform_test, used_data=['test'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)

    # train model
    model = base_model(class_num=config.class_num)
    
    dummy_input = torch.randn(1,3,112,112)
    network_writer.add_graph(model, (dummy_input, ), True)
    network_writer.close()

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1, last_epoch=-1)
    creiteron = torch.nn.CrossEntropyLoss()

    # you may need train_numbers and train_losses to visualize something
    train_numbers, train_losses = train(config, train_loader, model, optimizer, scheduler, creiteron)

    # you can use validation dataset to adjust hyper-parameters
    val_accuracy = test(val_loader, model)
    test_accuracy = test(test_loader, model)
    print('===========================')
    print("val accuracy:{}%".format(val_accuracy * 100))
    print("test accuracy:{}%".format(test_accuracy * 100))

    if not os.path.exists('model_list.csv'):
        with open('model_list.csv', "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model name', 'test acc', 'learning rate', 'epoch', 'batch size', 'is scheduler', 'scheduler method'])

    with open('model_list.csv', "a+", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['./model_'+exp_time+'.pth', str(test_accuracy), str(config.learning_rate),\
            str(config.epochs), str(config.batch_size), 'True', 'MultiStepLR: milestone'+str(config.milestones)])

    # draw
    get_features(config, transform_test, 'test')
    show_samples('test')
    get_features(config, transform_test, 'train')
    show_samples('train')


def train(config, data_loader, model, optimizer, scheduler, creiteron):
    model.train()
    model = model.cuda()
    train_losses = []
    train_numbers = []
    counter = 0
    for epoch in range(config.epochs):
        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.cuda(), label.cuda()
            output = model(data)
            loss = creiteron(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter += data.shape[0]
            accuracy = (label == output.argmax(dim=1)).sum() * 1.0 / output.shape[0]
            if batch_idx % 20 == 0:
                print('Train Epoch: {} / {} [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {:.6f}'.format(
                    epoch+1, config.epochs, batch_idx * len(data)+1, len(data_loader.dataset),
                                          100. * batch_idx / len(data_loader), loss.item(), accuracy.item()))
                train_loss_writer.add_scalar('training_loss', loss.item(), epoch*len(data_loader)+batch_idx)
                train_acc_writer.add_scalar('training_acc', accuracy.item(), epoch*len(data_loader)+batch_idx)
                train_losses.append(loss.item())
                train_numbers.append(counter)
        scheduler.step()
        torch.save(model.state_dict(), './weights/model_'+exp_time+'.pth')
    return train_numbers, train_losses


def test(data_loader, model):
    model.eval()
    model = model.cuda()
    correct = 0
    with torch.no_grad():
        for data, label in data_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum()
    accuracy = correct * 1.0 / len(data_loader.dataset)
    return accuracy


def get_features(config, transform, sample_set):

    show_dataset = tiny_caltech35(transform=transform, used_data=[sample_set])
    show_loader = DataLoader(show_dataset, batch_size=1, shuffle=False, drop_last=False)
    model = feature_model(class_num=config.class_num)
    model.load_state_dict(torch.load('./weights/model_'+exp_time+'.pth'))
    # model.load_state_dict(torch.load('./weights/model_20220510-21-56-35.pth'))
    model.eval()
    model.cuda()

    feature_list = torch.tensor([]).cuda()
    label_list = torch.tensor([]).cuda()
    
    with torch.no_grad():

        for data, label in tqdm(show_loader):
            data, label = data.cuda(), label.cuda()
            feature = model(data)
            feature_list = torch.cat((feature_list, feature), 0)
            label_list = torch.cat((label_list, label), 0)
    
    feature_list = feature_list.cpu()

    tsne = TSNE(n_components=2)
    tsne_feature = tsne.fit_transform(feature_list)

    np.save('./feature/tsne_'+exp_time+'_'+sample_set+'.npy', tsne_feature)
    np.save('./feature/label_'+exp_time+'_'+sample_set+'.npy', label_list.cpu())


def show_samples(sample_set):

    tsne_feature = np.load('./feature/tsne_'+exp_time+'_'+sample_set+'.npy')
    label_list = np.load('./feature/label_'+exp_time+'_'+sample_set+'.npy')

    label_list = label_list.astype('int')
    label_list = label_list.astype('str')

    label_str = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    for label_idx in range(7):
        label_list[label_list == str(label_idx)] = label_str[label_idx]
    
    sns.set(font_scale=0.5, )
    pac_feature_fig = sns.scatterplot(x=tsne_feature[:,0], y=tsne_feature[:,1], hue=label_list, s=4, alpha=1, palette='muted')
    scatter_fig = pac_feature_fig.get_figure()
    scatter_fig.savefig('./image/sample_image_'+exp_time+'_'+sample_set+'.png', dpi=400)
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, nargs='+', default=[112, 112])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--class_num', type=int, default=7)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--milestones', type=int, nargs='+', default=[20, 25])

    config = parser.parse_args()
    main(config)
    train_loss_writer.close()
    train_acc_writer.close()
