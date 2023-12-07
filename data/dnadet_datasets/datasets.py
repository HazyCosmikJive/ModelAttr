'''
dataset for DNA-Det - AAAI 2022
    Yang, Tianyun, et al. "Deepfake network architecture attribution." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 4. 2022.
ref: https://github.com/ICTMCG/DNA-Det
'''

import random
import numpy as np

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from .transforms import MultiCropTransform, get_transforms

def read_annotations(annotations):
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

class ImageDataset(Dataset):
    def __init__(self, annotations, config, balance=False):
        self.config = config
        self.balance = balance
        self.class_num=config.model.class_num
        self.resize_size = config.data.resize_size
        self.second_resize_size = None
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if config.data.get("label_map", None):
            self.label_map = config.data.label_map
        if balance:
            self.data = [[x for x in annotations if int(x[1]) == lab] for lab in [i for i in range(self.class_num)]]
        else:
            self.data = [annotations]

    def __len__(self):
        
        return max([len(subset) for subset in self.data])

    def __getitem__(self, index):
        
        if self.balance:
            labs = []
            imgs = []
            img_paths = []
            for i in range(self.class_num):
                safe_idx = index % len(self.data[i])
                img_path, lab = self.data[i][safe_idx]
                img = int(self.load_sample(img_path))
                labs.append(lab)
                imgs.append(img)
                img_paths.append(img_path)
                
            return torch.cat([imgs[i].unsqueeze(0) for i in range(self.class_num)]),\
                torch.tensor(labs, dtype=torch.long), img_paths
        else:
            img_path, lab = self.data[0][index]
            img = self.load_sample(img_path)
            lab = torch.tensor(int(lab), dtype=torch.long)
            
            return img, lab, img_path

    def load_sample(self, img_path):
        
        img = Image.open(img_path).convert('RGB')
        if img.size[0]!=img.size[1]:
            img = transforms.CenterCrop(size=self.config.data.get("crop_size", (64, 64)))(img)
        if self.resize_size is not None:
            img = img.resize(self.resize_size)
        if self.second_resize_size is not None:
            img = img.resize(self.second_resize_size)
        
        img = self.norm_transform(img)    

        return img


class ImageMultiCropDataset(ImageDataset):
    def __init__(self, annotations, config, balance=False, mode="train"):
        super(ImageMultiCropDataset, self).__init__(annotations, config, balance)
        
        self.multi_size = config.data.multi_size
        self.mode = mode
        crop_transforms = []
        for s in self.multi_size:
            RandomCrop = transforms.RandomCrop(size=s)
            crop_transforms.append(RandomCrop)
            self.multicroptransform = MultiCropTransform(crop_transforms)
            
    def __getitem__(self, index):
        #! currrently do not implement balance.
        if self.balance:
            labs = []
            imgs = []
            crops = []
            img_paths = []
            for i in range(self.class_num):
                safe_idx = index % len(self.data[i])
                img_path = self.data[i][safe_idx][0]
                img, crop = self.load_sample(img_path)
                lab = int(self.data[i][safe_idx][1])
                if hasattr(self, 'label_map'):
                    lab = self.label_map[lab]
                labs.append(lab)
                imgs.append(img)
                crops.append(crop)
                img_paths.append(img_path)
            crops = [torch.cat([crops[c][size].unsqueeze(0) for c in range(self.class_num)])
                for size in range(len(self.multi_size))]

            # return torch.cat([imgs[i].unsqueeze(0) for i in range(self.class_num)]),\
            #     crops, torch.tensor(labs, dtype=torch.long), img_paths
            return dict(
                imgs = torch.cat([imgs[i].unsqueeze(0) for i in range(self.class_num)]),
                img_paths = img_path,
                crops = crops,
                labels = torch.tensor(labs, dtype=torch.long)
            )
        else:
            img_path, lab = self.data[0][index][0], self.data[0][index][1]
            if hasattr(self, 'label_map'):
                lab = self.label_map[lab]
            lab = torch.tensor(int(lab), dtype=torch.long)
            img, crops = self.load_sample(img_path)

            # only return concated crops
            if self.mode == "train":
                img = torch.stack(crops, dim=0)
            return dict(
                imgs = img,
                img_paths = img_path,
                crops = crops,
                labels = lab
            )

    def load_sample(self, img_path):
        img = Image.open(img_path).convert('RGB')
        if img.size[0]!=img.size[1]:
            img = transforms.CenterCrop(size=self.config.data.crop_size)(img)

        if self.resize_size is not None:
            img = img.resize(self.resize_size)
        if self.second_resize_size is not None:
            img = img.resize(self.second_resize_size)
            
        crops = self.multicroptransform(img)
        img = self.norm_transform(img)
        crops = [self.norm_transform(crop) for crop in crops]

        return img, crops

class ImageTransformationDataset(ImageDataset):
    def __init__(self, annotations, config, balance=False):
        super(ImageTransformationDataset, self).__init__(annotations, config, balance)
    
        self.data = annotations
        self.pretrain_transforms = get_transforms(config.data.crop_size)
        self.class_num = self.pretrain_transforms.class_num
        crop_transforms = []
        self.multi_size = config.data.multi_size
        for s in self.multi_size:
            RandomCrop = transforms.RandomCrop(size=s)
            crop_transforms.append(RandomCrop)
            self.multicroptransform = MultiCropTransform(crop_transforms)
            
    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, index):
        
        img_path = self.data[index]
        if isinstance(img_path, tuple):
            img_path = img_path[0]
        img = Image.open(img_path).convert('RGB')
        
        # import ipdb
        # ipdb.set_trace()
        # img = transforms.RandomCrop(size=self.config.crop_size)(img)

        while True:
            img_path = self.data[index]
            if isinstance(img_path, tuple):
                img_path = img_path[0]
            img = Image.open(img_path).convert('RGB')
            if img.size[0] < self.config.data.crop_size[0] or img.size[1] < self.config.data.rop_size[1]:
                index = np.random.randint(0, len(self.data))
                print(">> rand another idx")
            else:
                break
        img = transforms.RandomCrop(size=self.config.data.crop_size)(img)
        
        select_id=random.randint(0,self.class_num-1)
        pretrain_transform=self.pretrain_transforms.select_tranform(select_id)
        transformed = pretrain_transform(image=np.asarray(img))
        img = Image.fromarray(transformed["image"])

        if self.resize_size is not None:
            img = img.resize(self.resize_size)

        crops = self.multicroptransform(img)
        img = self.norm_transform(img)
        crops = [self.norm_transform(crop) for crop in crops]
        lab = torch.tensor(select_id, dtype=torch.long)
    
        # return img, crops, lab, img_path
        return dict(
            imgs = img,
            img_paths = img_path,
            crops = crops,
            labels = lab
        )

class BaseData(object):
    def __init__(self, train_data_path, val_data_path, config):

        train_set = ImageDataset(read_annotations(train_data_path), config, balance=True)
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.trainer.workers,
            batch_size=config.trainer.batchsize,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )
        
        val_set = ImageDataset(read_annotations(val_data_path), config, balance=False)
        val_loader = DataLoader(
            dataset=val_set,
            num_workers=config.trainer.workers,
            batch_size=config.trainer.batchsize,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        print('train: {}, val: {}'.format(len(train_set),len(val_set)))


class SupConData(object):
    def __init__(self, train_data_path, val_data_path, config):
        
        train_set = ImageMultiCropDataset(read_annotations(train_data_path), config, balance=False, mode="train")
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.trainer.workers,
            batch_size=config.trainer.batchsize,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )

        val_set = ImageMultiCropDataset(read_annotations(val_data_path), config, balance=False, mode="test")
        val_loader = DataLoader(
            dataset=val_set,
            num_workers=config.trainer.workers,
            batch_size=config.trainer.batchsize,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        print('train: {}, val: {}'.format(len(train_set),len(val_set)))


class TranformData(object):
    def __init__(self, train_data_path, val_data_path, config):
        train_set = ImageTransformationDataset(read_annotations(train_data_path), config)
        import ipdb
        ipdb.set_trace()
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.trainer.workers,
            batch_size=config.trainer.batchsize,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
        self.train_loader = train_loader
        self.class_num = train_set.class_num

        val_set = ImageTransformationDataset(read_annotations(val_data_path), config)
        val_loader = DataLoader(
            dataset=val_set,
            num_workers=config.trainer.workers,
            batch_size=config.trainer.batchsize,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
        self.val_loader = val_loader
        
        print('train: {}, val: {}'.format(len(train_set),len(val_set)))
        

