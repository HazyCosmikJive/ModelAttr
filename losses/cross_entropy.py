import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, use_crops=False):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.use_crops = use_crops
    
    def forward(self, output, labels):
        logits = output["logits"]
        if self.use_crops == True: # for DNA-Det
            # for image labels [0, 1, 1], crop to 16 crops, 
            crop_nums = logits.shape[0] // labels.shape[0]
            crop_labels = labels.unsqueeze(1).expand(labels.shape[0], crop_nums).reshape(-1, 1).squeeze(1)
            loss = self.criterion(logits, crop_labels)
        elif len(logits.shape) == 4:
            # patch_cnn returned patch-level logits
            n, c, h, w = logits.shape
            patch_labels = labels.view(-1, 1, 1).expand(n, h, w)
            loss = self.criterion(logits, patch_labels)
        else:
            loss = self.criterion(logits, labels)
        # loss = self.criterion(logits, labels)
        return loss