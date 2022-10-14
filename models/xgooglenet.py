
import torch
import torch.nn as nn
import torchvision.models as models

class XGoogLeNet:
    
    def __init__(self, class_num, pre_trained, layer_idx):
        
        self.class_num = class_num
        self.pre_trained = pre_trained
        self.layer_idx = layer_idx
        self.model = models.googlenet(pretrained=self.pre_trained)
        for index, param in enumerate(self.model.parameters()):
            if index < self.layer_idx:
                param.requires_grad = False
        self.model.fc = nn.Linear(1024, self.class_num)