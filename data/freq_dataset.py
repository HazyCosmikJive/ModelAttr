import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image, ImageFilter
import cv2

from .datasets import ImageDataset
from utils.freq_transform import fft2d_tensor_wrapper, fft2d_tensor_wrapper_v2, dct2d_tensor_wrapper_v2


class FreqDataset(ImageDataset):
    def __init__(self, **kwargs):
        super(FreqDataset, self).__init__(**kwargs)
        self.freq_transform = self.config.data.get("freq_transform", "FFT")

    def load_sample(self, imgpath):
        image = Image.open(imgpath).convert("RGB")
        if self.gray:
            image = image.convert('L')
        try:
            image = transforms.RandomCrop(size=self.first_crop_size)(image)
        except:
            image = transforms.CenterCrop(size=self.first_crop_size)(image)
        image = image.resize(self.resize_size)
        image = self.transforms(image)
        if self.high_pass:
            filter = torch.tensor([[[[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]]]).to(image.device).float()
            image = F.conv2d(image.unsqueeze(1), filter, padding=1).squeeze(1)
        
        # transform to frequency
        if self.freq_transform == "FFT":
            # image_freq = fft2d_tensor_wrapper(image, self.config.data.get("scale_factor", 20))
            image_freq = fft2d_tensor_wrapper_v2(image, shift=self.config.data.get("shift", False), logscale_factor=self.config.data.get("logscale_factor", None))
        elif self.freq_transform == "DCT":
            image_freq = dct2d_tensor_wrapper_v2(image)
        else:
            raise NotImplementedError("Frequency transform method {} is not implemented".format(self.freq_transform))
        return image_freq