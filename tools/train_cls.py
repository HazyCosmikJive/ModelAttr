import sys

import torch
import torchvision.utils as vutils
import torch.distributed as dist

import mmcv
import datetime
from torch_utils import AverageMeter

from utils.checkpoint import save_checkpoint
from utils.dist import get_rank, get_world_size
from tools.evaluation import eval_model

import ipdb

# [tmp] check function, visualze some samples
def vis_grid_imgs(imgs, path):
    import torchvision.utils as vutils
    # imgs.shape = [16, 3, 64, 64]
    grid = vutils.make_grid(imgs, nrow=16)
    vutils.save_image(grid, path)


def train_model(
    model,
    config,
    device,
    criterions,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    test_loader,
    logger,
    writer=None,
    train_sampler=None,
    val_sampler=None,
    awl=None,
):

    no_improve_cnt = 0 # for early stop
    best_epoch = 0
    best_metric = 0
    best_test_metric = 0

    epoch_time_meter = AverageMeter(name="EpochTime", length=5)

    loss_nan_cnt = 0

    for epoch in range(1, config.trainer.epoch + 1):
        if train_sampler is not None and config.common.get("dist", False): # dist
            train_sampler.set_epoch(epoch)
        timer = mmcv.Timer()
        epoch_timer = mmcv.Timer()
        losses = AverageMeter(name="Train_loss", length=config.trainer.print_freq)
        accmeter = AverageMeter(name="Train_acc", length=config.trainer.print_freq)
        batch_time = AverageMeter(name="BatchTime", length=config.trainer.print_freq)

        model.train()
        if awl is not None:
            awl.train()
        
        for i, batch in enumerate(train_loader):
            imgs, labels = batch["imgs"], batch["labels"]
            batchsize = imgs.shape[0]
            imgs = imgs.reshape((-1, imgs.size(-3), imgs.size(-2), imgs.size(-1)))
            labels = labels.reshape((-1))
            imgs, labels = [item.to(device) for item in [imgs, labels]]

            optimizer.zero_grad()

            output = model(imgs)

            if "supcontrast" in criterions.keys():  # reshape feats for contrast loss
                out_dim = output["feats"].shape[-1]
                output["feats"] = output["feats"].reshape(batchsize, -1, out_dim)

            optimizer.zero_grad()
            loss_dict = dict()
            loss_weights = config.model.loss.get("weights", [1.0] * len(criterions))
            loss = 0
            assert len(loss_weights) == len(criterions.keys())

            for key in criterions.keys():
                loss_dict[key] = criterions[key](output, labels)

            if awl is not None:  # use auto weighted loss
                loss = awl(loss_dict)
            else:  # use given loss weight
                for idx, key in enumerate(criterions.keys()):
                    loss += loss_weights[idx] * loss_dict[key]
            if torch.isnan(loss):
                torch.nan_to_num(loss)
                # import sys
                logger.info("[ERROR] Loss is NaN now, check input, replace with 0 now")
                loss_nan_cnt += 1
                if loss_nan_cnt >= 100:
                    logger.info("[ERROR] Loss is NaN for 100 times, exit")
                    import sys
                    sys.exit(0)
                # sys.exit(0)
            losses.update(loss.item())
            loss.backward()

            # only for a better print
            for k, v in loss_dict.items():
                loss_dict[k] = round(v.item(), 4)

            optimizer.step()
            scheduler.step()

            if len(output["logits"].shape) == 4: # patch_cnn' patch-level predictions
                # vote
                preds = torch.mode(torch.argmax(output["logits"], dim=1).view(output["logits"].shape[0], -1))[0]
            else:
                preds = torch.argmax(output["logits"], dim=1)
                if len(preds) != len(labels): # DNA-Det, here we also take the vote results of patches as img-level prediction
                    preds = torch.mode(preds.view(batchsize, -1))[0]

            acc = sum(preds == labels) / preds.shape[0]
            accmeter.update(acc.item())

            if config.common.get("dist", False):
                # TODO: dist is not implemented now
                continue

            batch_time.update(timer.since_last_check())

            if i % config.trainer.print_freq == 0:
                eta_seconds = int(batch_time.avg * (len(train_loader) * (config.trainer.epoch - epoch + 1) - i))
                eta = str(datetime.timedelta(seconds=eta_seconds))
                logger.info("Epoch {} | TRAIN Iter [{}/{}] | Loss {:.4f} acc: {:.4f}| lr: {:.6f} batchtime: {:.4f} ETA: {} | Loss items: {}".format(
                    epoch,
                    i, len(train_loader),
                    losses.avg, accmeter.avg,
                    optimizer.param_groups[0]['lr'],
                    batch_time.avg,
                    eta,
                    loss_dict
                ))
                if writer is not None:
                    writer.add_train_scaler(
                        i,
                        epoch,
                        len(train_loader),
                        losses.avg,
                        loss_dict,
                        accmeter.avg,
                        optimizer.param_groups[0]['lr'],
                    )

        # eval
        with torch.no_grad():
            results = eval_model(model, config, epoch, device, val_loader, test_loader, logger, writer, val_sampler, test=True)
        val_result = results["val"]
        test_result = results["test"]
        if writer is not None:
            writer.add_val_metric_scaler(epoch, results["val"])
            writer.add_test_metric_scaler(epoch, results["test"])

        # save model
        if config.common.get("debug", False):  # debug mode
            model_savepath = config.common.checkpointpath_debug
        else:
            model_savepath = config.common.checkpointpath

        val_metric = val_result[config.trainer.get("metric", "acc")]
        test_metric = test_result[config.trainer.get("metric", "acc")]

        if get_rank() == 0:
            if epoch % config.trainer.get("save_freq", 1) == 0:
                # TODO: change optimizers and schedulers to dicts as above codes
                save_checkpoint(model_savepath, epoch, model, optimizer, scheduler, val_metric, best_metric, best_epoch, config)
            # update best model
            if val_metric >= best_metric:
                best_metric = val_metric
                best_test_metric = test_metric
                best_epoch = epoch
                no_improve_cnt = 0
                save_checkpoint(model_savepath, epoch, model, optimizer, scheduler, val_metric, best_metric, best_epoch, config, is_best=True)
            else:
                no_improve_cnt += 1

        epoch_time = epoch_timer.since_last_check()
        epoch_time_meter.update(epoch_time)
        endtime = str(datetime.datetime.now() + datetime.timedelta(seconds=(config.trainer.epoch - epoch) * epoch_time))
        epoch_time = datetime.timedelta(seconds=int(epoch_time_meter.avg))
        
        logger.info("\n===> Epoch {} \n     [Train] Loss {:.4f} acc {:.4f} \n     [Val] acc {:.4f} F1: {:.4f} Recall: {:.4f}\n     [Test] acc {:.4f} F1: {:.4f} Recall: {:.4f}\n     * Best val {}: {:.4f} at epoch {} with test {}: {:.4f}\n     Avarage epoch time: {}, estimated end time: {}".format(
                            epoch,
                            losses.avg, accmeter.avg,
                            val_result["acc"], val_result["f1"], val_result["recall"],
                            test_result["acc"], test_result["f1"], test_result["recall"],
                            config.trainer.get("metric", "acc"), best_metric, best_epoch, config.trainer.get("metric", "acc"), best_test_metric,
                            str(epoch_time), endtime,
                        ))

        if config.trainer.get("early_stop", False):
            if no_improve_cnt >= config.trainer.get("early_stop_bar", 10):
                logger.info("\nEarly Stop Here")
                sys.exit(0)
