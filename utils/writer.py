import numpy as np
import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter


def init_writer(config):
    writer = None
    if config.common.get("use_tensorlog", False):
        if config.common.get("debug", False):  # debug mode
            writer_dir = config.common.tblogpath_debug
        else:
            writer_dir = config.common.tblogpath
        writer = Summary(writer_dir=writer_dir, suffix=config.common.timestamp)
    return writer


class Summary(object):
    def __init__(self, writer_dir=None, suffix=None):
        if writer_dir:
            self.writer = SummaryWriter(writer_dir, filename_suffix=suffix)
        else:
            self.writer = SummaryWriter()

    # save train loss
    def add_train_scaler(self, i, epoch, epoch_len, loss, loss_dict, acc, lr):
        total_step = i + epoch * epoch_len
        self.writer.add_scalar('train/loss', loss, total_step)  # total loss
        self.writer.add_scalar('train/acc', acc, total_step)
        self.writer.add_scalar('train/lr', lr, total_step)

        # save each loss
        for key in loss_dict.keys():
            self.writer.add_scalar('train/losses_{}'.format(key), loss_dict[key], total_step)

    # save val loss
    def add_val_scaler(self, i, epoch, epoch_len, loss):
        total_step = i + epoch * epoch_len
        self.writer.add_scalar('val/g_loss', loss, total_step)

    def add_val_metric_scaler(self, epoch, val_result):
        # val_result.keys(): acc, recall, f1
        total_step = epoch
        self.writer.add_scalar('val/acc', val_result["acc"], total_step)
        self.writer.add_scalar('val/f1', val_result["f1"], total_step)
        self.writer.add_scalar('val/recall', val_result["recall"], total_step)

    # save test loss
    def add_test_scaler(self, i, epoch, epoch_len, loss):
        total_step = i + epoch * epoch_len
        self.writer.add_scalar('val/g_loss', loss, total_step)

    def add_test_metric_scaler(self, epoch, test_result):
        # test_result.keys(): acc, recall, f1
        total_step = epoch
        self.writer.add_scalar('test/acc', test_result["acc"], total_step)
        self.writer.add_scalar('test/f1', test_result["f1"], total_step)
        self.writer.add_scalar('test/recall', test_result["recall"], total_step)

    def add_histogram(self, root_name, model, i, epoch, epoch_len, grad=False):
        total_step = i + epoch * epoch_len
        if isinstance(model, list):
            model = np.stack(model, axis=0)
        if isinstance(model, torch.Tensor):
            model = model.clone().cpu().data.numpy()
        if isinstance(model, np.ndarray):
            self.writer.add_histogram(root_name + '/', model, total_step)
            return

        for name, param in model.named_parameters():
            self.writer.add_histogram(root_name + '/' + name,
                                      param.clone().cpu().data.numpy(),
                                      total_step)
            if grad:
                self.writer.add_histogram(
                    'grad/' + name,
                    param.grad.clone().cpu().data.numpy(), total_step)

    def add_graph(self, model, input_size):
        # if i == 0 and epoch == self.args.start_epoch:
        #     self.writer.add_graph(model, input_var)
        demo_input = torch.rand(input_size)
        self.writer.add_graph(model, demo_input)

    def add_image(self, name, frames, i, epoch, epoch_len):
        total_step = i + epoch * epoch_len

        x = frames.clone().cpu().data.numpy()
        self.writer.add_histogram(name + 'histogram', x, total_step)
        x = x.transpose(0, 2, 1, 3, 4)
        x = np.ascontiguousarray(x, dtype=np.float32)
        x = x.reshape(-1, *x.shape[-3:])

        grid = vutils.make_grid(torch.from_numpy(x), normalize=True)
        self.writer.add_image(name + 'image', grid, total_step)

    def close(self):
        self.writer.close()