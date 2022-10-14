
'''
output a dataloader
'''

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class XDataLoader:

    def __init__(self, root_path, is_train, img_size, batch_size):
        
        self.root_path = root_path
        self.is_train = is_train
        self.img_size = img_size
        self.batch_size = batch_size
        self.dataset = None
        self.transform = None
        self.dataloader = None
    
    
    def _data_transform(self):
    
        if self.is_train:
            self.transform = {
                            'train':transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(30),
                                transforms.Resize([self.img_size, self.img_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                ]),
                            'val':transforms.Compose([
                                transforms.Resize([self.img_size, self.img_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                ])
                            }
        else:
            self.transform = {x:transforms.Compose([
                                transforms.Resize([self.img_size, self.img_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                ])
                                for x in ['test']}
    
    
    def _make_dataset(self):
        
        if self.is_train:
            self.dataset = {x:datasets.ImageFolder(root = os.path.join(self.root_path,x),
                            transform=self.transform[x])
                            for x in ['train', 'val']}
        else:
            self.dataset = {x:datasets.ImageFolder(root = os.path.join(self.root_path,x),
                            transform=self.transform[x])
                            for x in ['test']}
    
    
    def _make_dataloader(self):
        
        if self.is_train:
            self.dataloader = {x:DataLoader(dataset = self.dataset[x],
                                batch_size=self.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=8)
                                for x in ['train', 'val']}
        else:
            self.dataloader = {x:DataLoader(dataset = self.dataset[x],
                                batch_size=1,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=8)
                                for x in ['test']}
    
    
    def load_data(self):
        
        self._data_transform()
        self._make_dataset()
        self._make_dataloader()