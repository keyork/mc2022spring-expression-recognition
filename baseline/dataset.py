import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torchvision.datasets.caltech


class tiny_caltech35(Dataset):
    def __init__(self, transform=None, used_data=['train']):
        self.train_dir = 'fer/train/'
        self.val_dir = 'fer/val/'
        self.test_dir = 'fer/test/'
        self.used_data = used_data
        for x in used_data:
            assert x in ['train', 'val', 'test']
        self.transform = transform

        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        self.samples, self.annotions = self._load_samples()

    def _load_samples_one_dir(self, dir='fer/train/'):
        samples, annotions = [], []

        sub_dir = os.listdir(dir)
        for i in sub_dir:
            tmp = os.listdir(os.path.join(dir, i))
            samples += [os.path.join(dir, i, x) for x in tmp]
            annotions += [self.classes.index(i)] * len(tmp)
        return samples, annotions

    def _load_samples(self):
        samples, annotions = [], []
        for i in self.used_data:
            if i == 'train':
                tmp_s, tmp_a = self._load_samples_one_dir(dir=self.train_dir)
            elif i == 'val':
                tmp_s, tmp_a = self._load_samples_one_dir(dir=self.val_dir)
            elif i == 'test':
                tmp_s, tmp_a = self._load_samples_one_dir(dir=self.test_dir)
            else:
                print('error used_data!!')
                exit(0)
            samples += tmp_s
            annotions += tmp_a
        return samples, annotions

    def __getitem__(self, index):
        img_path, img_label = self.samples[index], self.annotions[index]
        img = self._loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_label

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')

    def __len__(self):
        return len(self.samples)
