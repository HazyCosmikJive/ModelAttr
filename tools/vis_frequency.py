'''
Visualization of frequency spectrums, saved in class folders;
For each class, select some images to get average spectrum fig.

TODO: use residual for better visualization results
'''

import os
import cv2
import sys
import pprint
import itertools

import torch
from torch import nn as nn
import torchvision.utils as vutils

from data.dataset_entry import get_test_loader
from utils.log import setup_logger
from utils.parser import parse_cfg
from utils.init_config import init_config
from utils.dist import get_rank
from utils.freq_transform import fft2d_tensor_wrapper, dct2d_tensor_wrapper

from tqdm import tqdm

import ipdb

def main():
    arg = parse_cfg()
    config = init_config(arg, makedirs=False)

    logger = setup_logger(config)
    # logger.info("Config:\n {}".format(config.pretty_text))
    # savepath
    frequency_path = os.path.join(config.common.workspace, "frequency")
    
    os.makedirs(frequency_path, exist_ok=True)

    # set device (single GPU now)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    assert len(arg.gpu) == 1, "Single GPU now"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu[0]) # single gpu here
    device = torch.device("cuda:{}".format(str(arg.gpu[0])))

    # dataloader
    test_loader = get_test_loader(config, logger)

    # visualize a fix num of fingerprints for each class
    vis_per_class = config.vis_freq.get("vis_per_class", 200)
    logger.info("Visualizing frequency spectrums...")

    transform_method = config.vis_freq.get("freq_transform_method", "FFT")
    freq_results = []
    gt_labels = test_loader.dataset.get_gt_labels()
    CLASSES = config.data.CLASSES
    for class_idx, class_num in enumerate(CLASSES):
        os.makedirs(os.path.join(frequency_path, class_num), exist_ok=True)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            imgs, labels = batch["imgs"], batch["labels"]
            batchsize = imgs.shape[0]
            imgs = imgs.reshape((-1, 3, imgs.size(-2), imgs.size(-1)))
            labels = labels.reshape((-1))

            freq_results.extend(list(imgs.split(1, dim=0)))
                
    for class_idx in range(len(CLASSES)):
        label_mask = (gt_labels == class_idx)
        if label_mask.sum() == 0:
            continue
        imgs_selected = list(itertools.compress(freq_results, label_mask))
        class_tensor = torch.stack(imgs_selected, dim=0)
        mean_class = torch.einsum("chw->hwc", class_tensor.mean(dim=0).squeeze(0)).cpu().numpy()

        savepath = os.path.join(os.path.join(frequency_path, "{}_highpass_{}.png".format(CLASSES[class_idx], transform_method)))
        cv2.imwrite(savepath, mean_class)
        
        for img_idx, img in enumerate(imgs_selected[:vis_per_class]):
            savepath = os.path.join(os.path.join(frequency_path, CLASSES[class_idx], "img_{}_{}.png".format(img_idx, transform_method)))
            cv2.imwrite(savepath, torch.einsum("chw->hwc", imgs_selected[img_idx].squeeze(0)).cpu().numpy())
        logger.info("Save samples of class {}".format(CLASSES[class_idx]))
    logger.info("FINISH")


if __name__ == "__main__":
    main()