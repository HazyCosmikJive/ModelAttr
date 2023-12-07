'''
UniFD - CVPR2023:
    Ojha, Utkarsh, Yuheng Li, and Yong Jae Lee. "Towards universal fake image detectors that generalize across generative models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPModel
from models.heads import Predictor
from models.adapter import Adapter

class CLIPImageModel(nn.Module):
    '''
    huggingface implemented CLIP model
    '''
    def __init__(self, loadpath, adapter=False, freeze_all=True):
        super(CLIPImageModel, self).__init__()
        assert loadpath is not None, "Requires CLIP loadpath, url or filepath"
        self.model = CLIPModel.from_pretrained(loadpath)

        for parameter in self.model.parameters():
            parameter.requires_grad = False
                
        if adapter:
            in_channel = 512 if "ViT" in loadpath else 1024
            self.adapter = Adapter(c_in=in_channel)


    def forward(self, x):
        output = self.model.vision_model(x)
        image_feats = output.pooler_output
        all_feats = output.last_hidden_state
        if hasattr(self, "adapter"):
            x = self.adapter(image_feats)
            ratio = 0.2
            image_feats = ratio * x + (1 - ratio) * image_feats
        return image_feats, all_feats


class CLIPLinear(nn.Module):
    def __init__(self, config):
        '''
        fixed CLIP model + FC.

        Args in config:
        clip_loadpath (str): load path for huggingface CLIP
        adapter (bool): whether to use adapter for CLIP
            Default: False
        '''
        super().__init__()
        # fixed clip extractor
        self.clip_extractor = CLIPImageModel(config.model.get("clip_loadpath", None), config.model.get("adapter", False))
        self.fc = nn.Linear(768, config.model.class_num)

    def forward(self, x):
        cls_feats, patch_feats = self.clip_extractor(x)
        logits = self.fc(cls_feats)

        return dict(
            logits=logits,
            feats=F.normalize(cls_feats, dim=1),
            patch_feats=F.normalize(patch_feats, dim=1),
        )