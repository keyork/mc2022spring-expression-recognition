
import torch
import torch.nn as nn

class XDlibMLP(nn.Module):

    def __init__(self, num_classes: int = 7):
        super(XDlibMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.classifier(x)
        return x