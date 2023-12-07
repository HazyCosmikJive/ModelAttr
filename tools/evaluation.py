import torch
import torch.distributed as dist
import numpy as np

from torch_utils import AverageMeter

from utils.dist import get_rank, get_world_size
from utils.metric import evaluate_multiclass, confusion_matrix

def eval_model(
    model,
    config,
    epoch,
    device,
    val_loader,
    test_loader,
    logger,
    writer=None,
    val_sampler=None,
    test=False,
):
    
    model.eval()

    if val_sampler is not None:
        val_sampler.set_epoch(epoch)

    # ------------ val
    logger.info("--> VAL")

    preds = predict(model, val_loader, device, epoch, config, logger, writer)
    gt_labels = preds["gt_labels"]
    pred_labels = preds["pred_labels"]
    val_result = evaluate_multiclass(gt_labels, pred_labels)
    val_confusion_matrix = confusion_matrix(gt_labels, pred_labels)
    logger.info("\n===> Epoch {} Val Confusion Matrix\n {}" .format(epoch, (val_confusion_matrix)))

    result = dict(
        val=val_result
    )

    # ------------ test
    if test:
        logger.info("--> TEST")
        preds = predict(model, test_loader, device, epoch, config, logger, writer)
        test_gt_labels = preds["gt_labels"]
        test_pred_labels = preds["pred_labels"]
        test_result = evaluate_multiclass(test_gt_labels, test_pred_labels)
        test_confusion_matrix = confusion_matrix(test_gt_labels, test_pred_labels)
        logger.info("\n===> Epoch {} Test Confusion Matrix\n {}" .format(epoch, (test_confusion_matrix)))
        
        result.update({"test": test_result})
    else:
        # return dummy results
        result.update({"test": {
            "acc": -1,
            "f1": -1,
            "recall": -1,
        }})

    return result

def predict(model, val_loader, device, epoch, config, logger=None, writer=None):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    losses = AverageMeter(name="Val_loss", length=config.trainer.print_freq)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            imgs, img_paths, labels = batch["imgs"], batch["img_paths"], batch["labels"]
            imgs = imgs.reshape((-1, imgs.size(-3), imgs.size(-2), imgs.size(-1)))
            labels = labels.reshape((-1))
            imgs, labels = [item.to(device) for item in [imgs, labels]]

            if hasattr(model, "forward_test"): # for methods that requires different code for train and test
                cls_logits = model.forward_test(imgs)["logits"]
            else:
                cls_logits = model(imgs)["logits"]
            
            try:
                loss_cls = criterion(cls_logits, labels)
            except:
                # patch cnn cannot directly use torch.CE to compute loss
                # too lazy to modify the code... simply return dummy loss now
                loss_cls = torch.tensor(0.).to(cls_logits.device)

            losses.update(loss_cls.item())

            if i == 0:
                probs = cls_logits
                gt_labels = labels
                img_paths = img_paths
            else:
                probs = torch.cat([probs, cls_logits], dim=0)
                gt_labels = torch.cat([gt_labels, labels], dim=0)
                img_paths += img_paths

            if i % config.trainer.print_freq == 0:
                logger.info("Epoch {} | EVAL Iter [{}/{}] | Loss {:.4f}".format(epoch, i, len(val_loader), losses.avg))
                if writer is not None:
                    writer.add_val_scaler(i, epoch, len(val_loader), losses.avg)

    # distribution has now been tested for current code base
    # TODO: gather results from multi gpus here
    if config.common.get("dist", False):
        labels_gather = [torch.zeros_like(gt_labels)] * get_world_size()
        probs_gather = [torch.zeros_like(probs)] * get_world_size()
        dist.all_gather(labels_gather, gt_labels) # TODO: try to use dist.gather
        dist.all_gather(probs_gather, probs)
        
        gt_labels = labels_gather[0]
        probs = probs_gather[0]

    gt_labels = gt_labels.cpu().numpy()
    if len(probs.shape) == 4: # for patch-cnn
        pred_labels = torch.mode(torch.argmax(probs, dim=1).view(probs.shape[0], -1))[0]
        pred_labels = pred_labels.cpu().numpy()
    else: # previous other methods
        probs = probs.cpu().numpy()
        pred_labels = np.argmax(probs, axis=1)

    return dict(
        gt_labels=gt_labels,
        pred_labels=pred_labels,
        pred_probs=probs
    )