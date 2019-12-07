# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
import pdb


class AccuracyAngleRange(object):
    def __init__(self,):
        self.reset()

    def reset(self,):
        self.preds = []
        self.labels = []

    def __call__(self, preds, data):
        preds = preds[0]
        labels = data['angle_range_label'].numpy().tolist()
        for _pred, _label in zip(preds, labels):
            _pred = np.argmax(_pred.cpu().numpy())
            self.preds.append(_pred)
            self.labels.append(_label)

    def value(self,):
        return np.mean(np.array(self.preds) == np.array(self.labels))



class RunningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def __call__(self, label_preds, data):
        label_trues = data['label']
        for lt, lp in zip(label_trues, label_preds):
            lt = lt.cpu().numpy()
            lp = lp.cpu().numpy()
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def value(self):
        hist = self.confusion_matrix
        pre_class_iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        return pre_class_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def value(self):
        return self.avg
