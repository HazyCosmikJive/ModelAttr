import sys
import time
import numpy as np
import collections
import torch

import importlib
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


def evaluate_multiclass(gt_labels, pred_labels):
    acc = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels, average='macro')
    recall = recall_score(gt_labels, pred_labels, average='macro')
    recalls = recall_score(gt_labels, pred_labels, average=None)  # all recall values
    return {'acc':acc, 'f1':f1, 'acc':acc, 'recall':recall, 'recalls':recalls,}

def evaluate(gt_labels, pred_labels, scores):
    n = len(gt_labels)
    tn, fp, fn, tp = confusion_matrix(gt_labels, pred_labels).reshape(-1)
    assert((tn + fp + fn + tp) == n)

    auc = roc_auc_score(gt_labels, scores)
    ap = average_precision_score(gt_labels, scores)
    sen = float(tp) / (tp + fn)
    spe = float(tn) / (tn + fp)
    f1 = 2.0*sen*spe / (sen + spe)
    acc = float(tn + tp) / n
    return {'auc':auc, 'ap':ap, 'sen':sen, 'spe':spe, 'f1':f1, 'acc':acc}

def calculate_f1iou(pd: torch.Tensor, gt: torch.Tensor, th=0.5):
    SMOOTH = 1e-12
    gt = gt.long()
    tp, fp, fn, tn = smp.metrics.get_stats(pd, gt, mode="binary", threshold=th)
    tp, fp, fn, tn = tp.squeeze(1), fp.squeeze(1), fn.squeeze(1), tn.squeeze(1)
    intersection = (pd * gt).float().sum((1, 2))
    union = (pd + gt).float().sum((1, 2)) - intersection
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    iou = _iou_score(tp, fp, fn, tn, SMOOTH=SMOOTH)

    f1 = _fbeta_score(tp, fp, fn, tn, beta=1, SMOOTH=SMOOTH)

    return f1.numpy(), iou.numpy()

def _fbeta_score(tp, fp, fn, tn, beta=1, SMOOTH=0.0):
    beta_tp = (1 + beta**2) * tp
    beta_fn = (beta**2) * fn
    score = (beta_tp + SMOOTH) / (beta_tp + beta_fn + fp + SMOOTH)
    return score

def _iou_score(tp, fp, fn, tn, SMOOTH=0.0):
    return (tp + SMOOTH) / (tp + fp + fn + SMOOTH)


def _accuracy(tp, fp, fn, tn, SMOOTH=0.0):
    return (tp + tn + SMOOTH) / (tp + fp + fn + tn + SMOOTH)


def _sensitivity(tp, fp, fn, tn, SMOOTH=0.0):
    return (tp + + SMOOTH) / (tp + fn + + SMOOTH)


def _specificity(tp, fp, fn, tn, SMOOTH=0.0):
    return (tn + + SMOOTH) / (tn + fp + + SMOOTH)


def _balanced_accuracy(tp, fp, fn, tn):
    return (_sensitivity(tp, fp, fn, tn) + _specificity(tp, fp, fn, tn)) / 2


def _positive_predictive_value(tp, fp, fn, tn, SMOOTH=0.0):
    return (tp + + SMOOTH) / (tp + fp + + SMOOTH)


def _negative_predictive_value(tp, fp, fn, tn, SMOOTH=0.0):
    return (tn + + SMOOTH) / (tn + fn + + SMOOTH)


def _false_negative_rate(tp, fp, fn, tn, SMOOTH=0.0):
    return (fn + + SMOOTH) / (fn + tp + SMOOTH)


def _false_positive_rate(tp, fp, fn, tn, SMOOTH=0.0):
    return (fp + SMOOTH) / (fp + tn + SMOOTH)


def _false_discovery_rate(tp, fp, fn, tn):
    return 1 - _positive_predictive_value(tp, fp, fn, tn)


def _false_omission_rate(tp, fp, fn, tn):
    return 1 - _negative_predictive_value(tp, fp, fn, tn)


def _positive_likelihood_ratio(tp, fp, fn, tn):
    return _sensitivity(tp, fp, fn, tn) / _false_positive_rate(tp, fp, fn, tn)


def _negative_likelihood_ratio(tp, fp, fn, tn):
    return _false_negative_rate(tp, fp, fn, tn) / _specificity(tp, fp, fn, tn)

class Progbar(object):
    """Displays a progress bar.
    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)