import torch.nn as nn

from .cross_entropy import CrossEntropyLoss
from .supcon_loss import SupConLoss


def loss_entry(config, logger):
    loss_dict = dict()
    loss_types = config.model.loss.types
    if isinstance(loss_types, str):
        loss_dict[loss_types] = build_loss(loss_types, config)
    elif isinstance(loss_types, list):
        for loss_name in loss_types:
            loss_dict[loss_name] = build_loss(loss_name.lower(), config)
    logger.info("[LOSS] Use Loss: {}".format(loss_types))
    logger.info("[LOSS] Build Loss Done.")
    return loss_dict

def build_loss(loss_name, config):
    if "ce" == loss_name:
        return CrossEntropyLoss(config.model.loss.get("use_crops", False))
    elif "supcontrast" == loss_name:
        return SupConLoss()
    else:
        raise NotImplementedError("Loss {} is not implemented now.".format(loss_name))