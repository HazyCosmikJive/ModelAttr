'''
checkpoint related. 
ref: https://github.com/amazon-research/video-contrastive-learning/blob/main/utils/util.py
'''

import os
import torch
import glob

def load_checkpointpath(
    config,
    logger,
    model,
    optimizer=None,
    scheduler=None,
    testmode=False,
    resume_last=False,
    resume_best=False,
    path=None,
    pretrain=False,
):
    if resume_best and path is None:
        path = os.path.join(config.common.checkpointpath, "best.pth.tar")
    if resume_last and path is None:
        pathlist = glob.glob(config.common.checkpointpath + "/epoch*.pth.tar")
        # sort with epoch num, choose latest epoch to resume
        pathlist = sorted(pathlist, key=lambda name: int(name.split("/")[-1].split("_")[0][5:]))
        path = pathlist[-1]
    elif path is None:  # load from a given path when path is not None
        logger.info("No checkpoint path provided. Allow resume_best / resume_last or provided a checkpoint path.(｡O_O｡)")
        return model
    assert os.path.exists(path), "Checkpoint {} do not exist".format(path)


    checkpoint = torch.load(path, map_location='cpu')
    if not pretrain:
        # TODO: load optimizer
        logger.info("[LOAD MODEL] load from {}, epoch {}".format(path, checkpoint["epoch"]))
        # model.load_state_dict({re.sub("^module.", "", k): v for k, v in checkpoint["model_dict"].items()})
        model.load_state_dict(checkpoint["model_dict"])
        logger.info("- model dict loaded")
        
        if not testmode:
            if optimizer is not None:
                pass # TODO
            if scheduler is not None:
                pass
        del checkpoint
    else:
        # load pretrain; only load matched keys
        load_and_match_pretrain(model, checkpoint, logger)

def save_checkpoint(model_savepath, epoch, model, optimizer, scheduer, val_metric, best_metric, best_epoch, config, is_best=False):
    if is_best:
        torch.save({
                "epoch": epoch,
                "model_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduer.state_dict(),
                "val_metric": val_metric,
                "best_metric": best_metric,
                "best_epoch": best_epoch
            }, os.path.join(model_savepath, "best.pth.tar"))
    else:
        torch.save({
                "epoch": epoch,
                "model_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduer.state_dict(),
                "val_metric": val_metric,
                "best_metric": best_metric,
                "best_epoch": best_epoch
        }, os.path.join(model_savepath, "epoch{}_{}_{:.4f}.pth.tar".format(epoch, config.trainer.get("metric", "acc"), val_metric)))


def load_and_match_pretrain(model, pretrain_dict, logger):
    '''
    load pretrain model's dict, find missing keys and unexpected keys.
    ref: https://github.com/facebookresearch/SlowFast/blob/HEAD/slowfast/utils/checkpoint.py
    '''
    pretrain_dict_match = {}
    not_used_layers = []

    for k, v in pretrain_dict["model_dict"].items():
        #! for dist training saved checkpoint
        if "module." in k:
            k = k.replace("module.", "")
        if k in model.state_dict().keys():
            if v.size() == model.state_dict()[k].size():
                pretrain_dict_match[k] = v
            else:
                not_used_layers.append(k)
        else:
            not_used_layers.append(k)
    
    # weights that do no have match from pretrain
    not_load_layers = [
        k
        for k in model.state_dict().keys()
        if k not in pretrain_dict_match.keys()
    ]

    # load pretraine weights
    missing_keys, unexpected_keys = model.load_state_dict(
        pretrain_dict_match, strict=False
    )
    logger.info("Missing keys: \n{}".format(missing_keys))
    logger.info("Unexpected keys: \n{}".format(unexpected_keys))