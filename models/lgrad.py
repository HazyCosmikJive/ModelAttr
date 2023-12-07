'''
LGrad - CVPR2023
    Tan, Chuangchuang, et al. "Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
ref: https://github.com/chuangchuangtan/LGrad/blob/HEAD/img2gad_pytorch/gen_imggrad.py#L76
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.classifiers import BasicClassifier
from segmentation_models_pytorch.encoders import get_encoder


#! [!!! bug alert !!!]
# there may exists bugs in this implementation.
# please use https://github.com/chuangchuangtan/LGrad/blob/HEAD/img2gad_pytorch/gen_imggrad.py to save grad input in advance.
class LGrad(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder= get_encoder(config.model.encoder, in_channels=config.model.get("in_channels", 3), weights="imagenet")
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.classifier = BasicClassifier(config)

    def set_requires_grad(self, x, flag=True):
        if isinstance(x, nn.Module):
            for p in x.parameters():
                p.requires_grad = True
        elif isinstance(x, torch.Tensor):
            x.requires_grad = True

    def forward(self, x):
        with torch.enable_grad():
            self.set_requires_grad(x)
            self.set_requires_grad(self.encoder)
            feats = self.encoder(x)[-1]
            grad = torch.autograd.grad(feats.sum(), x, create_graph=True, retain_graph=True, allow_unused=False)[0]
        if not torch.is_grad_enabled(): # during test
            self.set_requires_grad(self, False)

        cls_output = self.classifier(grad)
        return cls_output