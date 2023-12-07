import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class ConvAdapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(ConvAdapter, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c_in, c_in // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in // 4, c_in, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.convs(x)
        return x
    