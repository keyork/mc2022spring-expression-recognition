import torch.nn as nn


class base_model(nn.Module):
    def __init__(self, class_num=7):
        super(base_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), padding=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.GAP = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, class_num)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pooling(x)
        x = self.relu(self.conv2(x))
        x = self.max_pooling(x)
        x = self.conv3(x)
        x = self.GAP(x).squeeze(dim=3).squeeze(dim=2)
        # you can see this x as the feature, and use it to visualize something

        x = self.fc1(self.relu(x))
        x = self.fc2(self.relu(x))
        return x

class feature_model(nn.Module):
    def __init__(self, class_num=7):
        super(feature_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), padding=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.GAP = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, class_num)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pooling(x)
        x = self.relu(self.conv2(x))
        x = self.max_pooling(x)
        x = self.conv3(x)
        x = self.GAP(x).squeeze(dim=3).squeeze(dim=2)
        # you can see this x as the feature, and use it to visualize something

        # x = self.fc1(self.relu(x))
        # x = self.fc2(self.relu(x))
        return x
