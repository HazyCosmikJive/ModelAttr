import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter

from .transforms import get_train_transforms, get_val_transforms
import mmcv

class ImageDataset(Dataset):
    def __init__(
        self,
        config,
        annotations,
        mode="train",
        label_map=None,
        high_pass=False,
        low_pass=False,
        gray=False,
    ):
        self.config = config

        self.first_crop_size = self.config.data.transform.first_crop_size
        self.resize_size = self.config.data.transform.resize_size
        self.mode = mode

        if label_map is not None:
            self.label_map = label_map
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.gray = gray

        self.infolist = self.read_annotations(annotations)

        if config.data.get("transform", None) is not None:
            self.aug_transform = get_train_transforms(config)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])


    def __getitem__(self, index):
        lab = self.infolist[index][1]
        if hasattr(self, 'label_map'):
            lab = self.label_map[lab]
        lab = torch.tensor(int(lab), dtype=torch.long)

        # imgpath = self.infolist[index][0]
        # img = self.load_sample(imgpath)

        if self.mode == "train":
            while True:
                imgpath = self.infolist[index][0]
                try:
                    img = self.load_sample(imgpath)
                    break
                except:
                    # print("Failed to load image {}".format(index))
                    index = np.random.randint(0, len(self.infolist))
        else:
            imgpath = self.infolist[index][0]
            img = self.load_sample(imgpath)

        results = {
            "imgs": img,
            "img_paths": imgpath,
            "labels": lab
        }

        vae_labelmap = self.config.data.get("vae_labelmap", None)
        if vae_labelmap is not None:
            vae_lab = vae_labelmap[self.infolist[index][1]]
            vae_lab = torch.tensor(int(vae_lab), dtype=torch.long)
            results.update({"vae_labels": vae_lab})

        return results


    def load_sample(self, imgpath):
        image = Image.open(imgpath).convert("RGB")
        if self.gray:
            image = image.convert('L')
        if self.high_pass and False:  # deprecated
            image = image.filter(ImageFilter.FIND_EDGES)
        if self.low_pass:
            image = image.filter(ImageFilter.BLUR)
        # try:
        #     image = transforms.RandomCrop(self.first_crop_size)(image)
        # except:
        #     image = transforms.CenterCrop(self.first_crop_size)(image)
        if self.first_crop_size is not None:
            if self.mode == "train":
                image = transforms.RandomCrop(self.first_crop_size)(image)
            else:
                image = transforms.CenterCrop(self.first_crop_size)(image)
        if self.resize_size is not None:
            image = image.resize(self.resize_size)

        if hasattr(self, 'color_aug'):
            image = self.color_aug(image)

        #! TODO: better ways for augs
        if len(getattr(self, 'aug_transform', [])) > 0:
            image = np.array(image) # to cv2 img first
            image = self.aug_transform(image=image)["image"]
            image = Image.fromarray(image)

        image = self.transforms(image)

        #! filter high pass
        if self.high_pass:
            filter = torch.tensor([[[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]]).to(image.device).float()
            channels = image.shape[0]
            filter = filter.expand(channels, 1, -1, -1)
            # image = F.conv2d(image.unsqueeze(0), filter, padding=1).squeeze(0) # only gray imgs
            image = torch.cat([F.conv2d(image.unsqueeze(0)[:, i:i+1, :, :], filter[i:i+1, :, :], padding=1) for i in range(channels)]).squeeze(1) # for color imgs
            # print(image.shape)

        return image

    def read_annotations(self, annotations):
        if isinstance(annotations, str):
            with open(annotations) as f:
                samples = [x.strip().split('\t') for x in f.readlines()]
        elif isinstance(annotations, list):
            samples = []
            for path in annotations:
                with open(path) as f:
                    sample = [x.strip().split('\t') for x in f.readlines()]
                samples.extend(sample) 
        return samples

    def get_gt_labels(self):
        gt_labels = [info[1] for info in self.infolist]
        if hasattr(self, "label_map"):
            try:
                gt_labels = [self.label_map[lab] for lab in gt_labels]
            except:
                print("")
        gt_labels = np.array([int(lab) for lab in gt_labels])
        return gt_labels

    def __len__(self):
        return len(self.infolist)
