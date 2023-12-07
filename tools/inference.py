import os
import torch
import torch.distributed as dist
import numpy as np
from matplotlib import pyplot as plt

import mmcv
import pprint
from mlxtend.plotting import plot_confusion_matrix

from tools.evaluation import predict
from utils.checkpoint import load_checkpointpath
from utils.dist import get_rank, get_world_size
from utils.metric import evaluate_multiclass, confusion_matrix

def inference(
        model,
        config,
        device,
        test_loader,
        logger,
        ckptpath=None
    ):
    if ckptpath is None: # load last ckpt by default
        # load_checkpointpath(config, logger, model, testmode=True, resume_best=True)
        # epoch = "BEST"
        load_checkpointpath(config, logger, model, testmode=True, resume_last=True)
        epoch = "LAST"
    else:
        load_checkpointpath(config, logger, model, testmode=True, path=ckptpath)
        epoch = ckptpath.split("/")[-1].split("_")[0][5:]

    logger.info("--> TEST")
    model.eval()
    preds = predict(model, test_loader, device, epoch, config, logger)
    test_gt_labels = preds["gt_labels"]
    test_pred_labels = preds["pred_labels"]
    test_result = evaluate_multiclass(test_gt_labels, test_pred_labels)
    test_confusion_matrix = confusion_matrix(test_gt_labels, test_pred_labels)

    test_result.update({"confusion_matrix": test_confusion_matrix})
    
    # save results to pkl file
    savepath = os.path.join(config.common.predpath, config.common.test_tag + ".pkl")
    mmcv.dump(preds, savepath)
    logger.info("Saved pred results to {}".format(savepath))

    # save metric.txt
    savepath = os.path.join(config.common.predpath, config.common.test_tag + "_metric.txt")
    with open(savepath, 'w') as f:
        for k, v in test_result.items():
            print(f'{k}:\n{v}\n', file=f)
    f.close()

    # save confusion matrix fig
    savepath = os.path.join(config.common.predpath, config.common.test_tag + "_confusion_matrix.png")
    classes = config.data.get("CLASSES", None)
    # classes = None #! tmp remove all class labels.
    try:
        fig, ax = plot_confusion_matrix(conf_mat=test_confusion_matrix, hide_ticks=True, cmap=plt.cm.Blues, class_names=classes)
    except:
        fig, ax = plot_confusion_matrix(conf_mat=test_confusion_matrix, hide_ticks=True, cmap=plt.cm.Blues)
    plt.savefig(savepath)


    logger.info("\n===> Epoch {} Test Confusion Matrix\n {} \nacc: {} \nf1: {} \nrecall: {}".format(
        epoch, test_confusion_matrix, test_result["acc"], test_result["f1"], test_result["recall"]
    ))
    logger.info("\nTest Metrics for:\n{}".format(config.data.test_meta))
    logger.info("test_tag: {}\n".format(config.common.test_tag))
