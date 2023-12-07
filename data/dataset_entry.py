import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image

from .datasets import ImageDataset
from .dnadet_datasets.datasets import SupConData
from .freq_dataset import FreqDataset

from .sampler import *
from utils.dist import get_rank, get_world_size


def get_sampler(dataset, config, logger):
    sampler_config = config.data.get("sampler", None)
    if sampler_config is None:
        return None
    elif config.common.get("dist", False):
        sampler = torch.utils.data.DistributedSampler(dataset)
        logger.info("[DATA SAMPLER] Use dist sampler.")
        return sampler
    else:
        if sampler_config.name in ["balance"]:
            sampler = BalanceSampler(dataset, sampler_config.get("class_weight", None))
            label_cnt = sampler.count(dataset.infolist)

            logger.info("[DATA SAMPLER] Use {} sampler.".format(sampler_config.name))
            logger.info("[DATA SAMPLER] Class Distribution:")
            for label, count in label_cnt.items():
                logger.info("class {}: {}".format(label, count))
            return sampler
        else:
            raise NotImplementedError("Sampler {} is not implemented now.".format(sampler_config.name))

def get_train_val_loader(config, logger):
    dataset_type = config.data.get("dataset_type", "image_dataset")

    # train dataset
    if dataset_type == "image_dataset":
        train_set = ImageDataset(
            config=config,
            annotations=config.data.train_meta,
            mode="train",
            label_map=config.data.get("label_map", None),
            high_pass=config.data.get("high_pass", False),
            low_pass=config.data.get("low_pass", False),
        )
    elif dataset_type == "dnadet_dataset":
        Data = SupConData(config.data.train_meta, config.data.val_meta, config)
        return {
            "train_loader": Data.train_loader,
            "val_loader": Data.val_loader
        }
    elif dataset_type == "freq_dataset":
        train_set = FreqDataset(
            config=config,
            annotations=config.data.train_meta,
            mode="train",
            label_map=config.data.get("label_map", None),
            high_pass=config.data.get("high_pass", False),
            gray=config.data.get("gray", False),
        )
    else:
        raise NotImplementedError("Dataset {} is not implemented now.".format(dataset_type))
    
    # val dataset
    if dataset_type == "freq_dataset":
        val_set = FreqDataset(
            config=config,
            annotations=config.data.val_meta,
            mode="test",
            label_map=config.data.get("label_map", None),
            high_pass=config.data.get("high_pass", False),
            gray=config.data.get("gray", False),
        )
    else:
        val_set = ImageDataset(
            config=config,
            annotations=config.data.val_meta,
            mode="test",
            label_map=config.data.get("label_map", None),
            high_pass=config.data.get("high_pass", False),
            low_pass=config.data.get("low_pass", False),
        )

    logger.info("[DATA INFO] Trainset: {} | Valset: {}".format(len(train_set.infolist), len(val_set.infolist)))

    train_sampler = get_sampler(train_set, config, logger)
    val_sampler = None
    if config.common.get("dist", False):
        val_sampler = torch.utils.data.DistributedSampler(val_set)

    train_loader = DataLoader(
        train_set,
        batch_size=config.trainer.batchsize,
        num_workers=config.trainer.workers,
        pin_memory=True,
        shuffle=True if train_sampler is None else False,
        drop_last=True,
        sampler=train_sampler,
        persistent_workers=True if config.trainer.workers > 0 else False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.trainer.batchsize,
        num_workers=config.trainer.workers,
        pin_memory=True,
        shuffle=False,
        sampler=val_sampler,
        persistent_workers=True if config.trainer.workers > 0 else False,
    )      
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_sampler": train_sampler,
        "val_sampler": val_sampler,
    }

def get_test_loader(config, logger, enable_labelmap=True):
    dataset_type = config.data.get("dataset_type", "image_dataset")
    if dataset_type == "dnadet_dataset":
        from .dnadet_datasets.datasets import ImageMultiCropDataset, read_annotations
        test_set = ImageMultiCropDataset(read_annotations(config.data.test_meta), config, balance=False, mode="test")
        test_loader = DataLoader(
            dataset=test_set,
            num_workers=config.trainer.workers,
            batch_size=config.trainer.batchsize,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )
        return test_loader
    elif dataset_type == "freq_dataset":
        test_set = FreqDataset(
            config=config,
            annotations=config.data.test_meta,
            mode="test",
            label_map=config.data.get("label_map", None) if enable_labelmap else None,
            high_pass=config.data.get("high_pass", False),
            gray=config.data.get("gray", False),
        )
    else:
        test_set = ImageDataset(
            config=config,
            annotations=config.data.test_meta,
            mode="test",
            label_map=config.data.get("label_map", None) if enable_labelmap else None,
            gray=config.data.get("gray", False),
            high_pass=config.data.get("high_pass", False),
            low_pass=config.data.get("low_pass", False),
        )

    logger.info("[DATA INFO] Testset: {}".format(len(test_set.infolist)))

    test_sampler = None
    if config.common.get("dist", False):
        test_sampler = torch.utils.data.DistributedSampler(test_set)


    test_loader = DataLoader(
        test_set,
        batch_size=config.trainer.get("batchsize", 8),
        num_workers=config.trainer.workers,
        pin_memory=True,
        shuffle=False,
        sampler=test_sampler,
        persistent_workers=True if config.trainer.workers > 0 else False,
    )

    return test_loader
        