import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import ClassificationHead
from models.heads import Predictor


class BasicClassifier(nn.Module):
    '''
    Simplest backbone + FC classifier.
    Basic classifier for later modifications.

    Args in config:
        layer_idx (int): returned the layer_idx-th layer feature for later use.
            Default: -1
        imagenet_pretrain (bool): initialize with imagenet-pretrained
            Default: True
        in_channels (int): input channel of the model. May be changed for other format inputs.
            Default: 3

    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        layer_idx = self.config.model.get('layer_idx', -1)
        imagenet_pretrain = config.model.get("imagenet_pretrain", True)
        self.encoder = get_encoder(config.model.encoder, in_channels=config.model.get("in_channels", 3), weights="imagenet" if imagenet_pretrain else None)
        self.clshead = ClassificationHead(self.encoder.out_channels[layer_idx], config.model.class_num)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # return which layer's feats
        layer_idx = self.config.model.get('layer_idx', -1)
        feats = self.encoder(x)[layer_idx]
        unpooled_feats = feats
        logits = self.clshead(feats)
        feats = self.pool(feats).reshape(feats.shape[0], -1)
        return dict(
            logits=F.softmax(logits, dim=-1),
            feats=feats,
            unpooled_feats=unpooled_feats,
        )


class MLPClassifier(nn.Module):
    '''
    backbone + MLP projector + FC
    Add MLP projector for better classification results.

    Args in config:
        layer_idx (int): returned the layer_idx-th layer feature for later use.
            Default: -1
        imagenet_pretrain (bool): initialize with imagenet-pretrained
            Default: True
        in_channels (int): input channel of the model. May be changed for other format inputs.
            Default: 3
        proj_dim_mid (int): dimension of mid channel of the MLP projector.
            Default: 512
        proj_dim_out (int): dimension of output channle of the MLP projector.
            Default: 128
    '''
    def __init__(self, config):
        super().__init__()
        self.classifier = BasicClassifier(config)
        in_channels = self.classifier.encoder.out_channels[-1]
        self.proj_head = Predictor(in_channels, config.model.get("proj_dim_mid", 512), config.model.get("proj_dim_out", 128))  # Linear + BN + ReLU + Linear
        self.fc = nn.Linear(config.model.get("proj_dim_out", 128), config.model.class_num)

    def forward(self, x):
        classifier_output = self.classifier(x)
        z = classifier_output["feats"]
        z = z.view(z.shape[0], -1)
        z = self.proj_head(z)
        logits = self.fc(z)

        return dict(
            logits=logits,
            feats=z,
        )

