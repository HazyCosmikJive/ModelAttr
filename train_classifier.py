import os
import pprint
import numpy as np
import random
import shutil

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.checkpoint import load_checkpointpath

from utils.log import setup_logger
from utils.scheduler import get_scheduler
from utils.parser import parse_cfg
from utils.writer import init_writer
from utils.init_config import init_config
from models.model_entry import model_entry
from data.dataset_entry import get_test_loader, get_train_val_loader
from losses.loss_entry import loss_entry
from losses.autoweight_loss import AutomaticWeightedLoss
from tools.train_cls import train_model
from tools.inference import inference

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def main():
    arg = parse_cfg()
    config = init_config(arg)

    seed_everything(config.common.get("random_seed", 42))

    logger = setup_logger(config)
    logger.info("Config:\n {}".format(config.pretty_text))
    # copy current config to workspace
    configname = config.common.exp_tag + ".py"
    config.dump(os.path.join(config.common.workspace, configname))

    # set device (only support single GPU now)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    assert len(arg.gpu) == 1, "Single GPU now"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu[0]) # single gpu here
    device = torch.device("cuda:{}".format(str(arg.gpu[0])))

    # tensorboard
    writer = init_writer(config)
    
    # dataloader
    loader_dict = get_train_val_loader(config, logger)
    train_loader, val_loader = loader_dict["train_loader"], loader_dict["val_loader"]

    test_loader = get_test_loader(config, logger)

    # build model
    model = model_entry(config, logger)
    model = model.to(device)

    # load pretrain
    pretrain_path = config.model.get("pretrain", "")
    if pretrain_path != "":
        load_checkpointpath(config, logger, model, path=pretrain_path, pretrain=True)
        logger.info("[MODEL] Pretrain model from {}".format(pretrain_path))

    # loss
    criterions = loss_entry(config, logger)
    for k, v in criterions.items():
        criterions[k] = criterions[k].to(device)

    if config.model.get("awl", False):
        awl = AutomaticWeightedLoss(len(criterions.keys()))
        awl.to(device)
    else:
        awl = None

    # optimizer
    if config.trainer.optimizer.type == "AdamW":
        params = [
            {"params": model.parameters(), "lr": config.trainer.optimizer.lr},
        ]
        if awl is not None:
            params.append({"params": awl.parameters(), "lr": 0.01})
        optimizer= torch.optim.AdamW(params)
    else:
        raise NotImplementedError("Not implemented optimizer {}".format(config.trainer.optimizer.type))

    scheduler = get_scheduler(optimizer, len(train_loader), config)

    # start training
    logger.info("\n----------------- START TRAINING -----------------")
    train_model(model, config, device, criterions, optimizer, scheduler, train_loader, val_loader, test_loader, logger, writer, awl=awl)


if __name__ == "__main__":
    main()