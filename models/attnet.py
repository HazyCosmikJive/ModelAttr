'''
AttNet - ICCV2019
    Yu, Ning, Larry S. Davis, and Mario Fritz. "Attributing fake images to gans: Learning and analyzing gan fingerprints." Proceedings of the IEEE/CVF international conference on computer vision. 2019.
ref: https://github.com/ningyu1991/GANFingerprints
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttNet_formal(nn.Module):
    def __init__(self, config):
        num_classes = config.model.class_num
    
        super(AttNet_formal, self).__init__()
        self.net = nn.Sequential(
            # down 1
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),  #128，16
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, stride=2), #64，32 #layer6
            # down 2
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, stride=2), #32,64 #layer13
            # down 3
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, stride=2), #16,128 #layer20
            # down 4
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, stride=2), #8, 256 #layer27
            # down 5
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2), #layer33
            nn.AvgPool2d(2, stride=2), #4, 512 #layer34
            # last conv
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(512, num_classes)

    def forward(self, input_x):
        x = self.net(input_x)
        x = self.avgpool(x) #1,512
        feature = x.view(x.size(0), -1)
        out = self.dense(feature)
        out = F.softmax(out)
        return dict(
            logits=out,
            feats=feature,
        )