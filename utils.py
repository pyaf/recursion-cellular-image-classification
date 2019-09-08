import os
import pdb
import cv2
import time
import json
import torch
import random
import scipy
import logging
import traceback
import numpy as np
from datetime import datetime

# from config import HOME
from tensorboard_logger import log_value, log_images
from torchnet.meter import ConfusionMeter
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import cohen_kappa_score
from pycm import ConfusionMatrix
from extras import *


class CM(ConfusionMatrix):
    def __init__(self, *args):
        ConfusionMatrix.__init__(self, *args)

    def save(self, name, qwk, loss, **kwargs):
        """ add `qwk` and `loss` to the saved obj,
        Use json.load(fileobject) for reading qwk and loss values,
        they won't be read by ConfusionMatrix class
        """
        status = self.save_obj(name, **kwargs)
        obj_full_path = status["Message"]
        with open(obj_full_path, "r") as f:
            dump_dict = json.load(f)
            dump_dict["qwk"] = qwk
            dump_dict["loss"] = loss
        json.dump(dump_dict, open(obj_full_path, "w"))


def to_multi_label(target, classes):
    """[0, 0, 1, 0] to [1, 1, 1, 0]"""
    multi_label = np.zeros((len(target), classes))
    for i in range(len(target)):
        j = target[i] + 1
        multi_label[i][:j] = 1
    return np.array(multi_label)


def get_preds(arr, num_cls):
    """ takes in thresholded predictions (num_samples, num_cls) and returns (num_samples,)
    [3], arr needs to be a numpy array, NOT torch tensor"""
    mask = arr == 0
    # pdb.set_trace()
    return np.clip(np.where(mask.any(1), mask.argmax(1), num_cls) - 1, 0, num_cls - 1)


def compute_score_inv(thresholds, predictions, targets):
    predictions = predict(predictions, thresholds)
    score = cohen_kappa_score(predictions, targets, weights="quadratic")
    return 1 - score


class Meter:
    def __init__(self, phase, epoch, save_folder):
        self.predictions = []
        self.targets = []
        self.phase = phase
        self.epoch = epoch
        self.save_folder = os.path.join(save_folder, "logs")
        self.base_th = 0.5 #, 0.5, 0.5, 0.5, 0.5]  #

    def update(self, targets, outputs):
        """targets, outputs are detached CUDA tensors"""
        # get multi-label to single label
        #pdb.set_trace()
        targets = (torch.sum(targets, 1) - 1).numpy().astype('uint8')
        outputs = torch.sigmoid(outputs)
        outputs = (outputs > self.base_th).cpu().numpy().astype('uint8')
        outputs = get_preds(outputs, 5)#.astype('uint')
        # outputs = torch.sum((outputs > 0.5), 1) - 1

        #pdb.set_trace()
        self.targets.extend(targets.tolist())
        self.predictions.extend(outputs.tolist())
        # self.predictions.extend(torch.argmax(outputs, dim=1).tolist()) #[2]

    def get_best_thresholds(self):
        """Epoch over, let's get targets in np array [6]"""
        self.targets = np.array(self.targets)
        self.predictions = np.array(self.predictions)
        """ not using this function anymore """
        return self.base_th

        if self.phase == "train":
            return self.base_th

        """Used in the val phase of iteration, see [4]"""
        self.predictions = np.array(self.predictions)
        simplex = scipy.optimize.minimize(
            compute_score_inv,
            self.base_th,
            args=(self.predictions, self.targets),
            method="nelder-mead",
        )
        self.best_th = simplex["x"]
        print("Best thresholds: %s" % self.best_th)
        return self.best_th

    def get_cm(self):
        # pdb.set_trace()
        base_qwk = cohen_kappa_score(self.targets, self.predictions, weights="quadratic")
        base_cm = CM(self.targets, self.predictions)
        return base_cm, base_qwk
        """ not used """
        if self.phase != "train":
            best_preds = predict(self.predictions, self.best_th)
            best_qwk = cohen_kappa_score(self.targets, best_preds, weights="quadratic")
            best_cm = CM(self.targets, best_preds)
            return base_cm, base_qwk, best_cm, best_qwk


def epoch_log(opt, log, tb, phase, epoch, epoch_loss, meter, start):
    base_cm, base_qwk = meter.get_cm()
    """
    if phase == "train":
        base_cm, base_qwk = meter.get_cm()
    else:
        base_cm, base_qwk, best_cm, best_qwk = meter.get_cm()
        # take care of best metrics
        acc, tpr, ppv, cls_tpr, cls_ppv = get_stats(best_cm)
        metrics = [acc, best_qwk, tpr, ppv]
        log_metrics(tb[phase], metrics, epoch, "best")
        log("best: QWK: %0.4f | ACC: %0.4f | TPR: %0.4f | PPV: %0.4f"
            % (best_qwk, acc, tpr, ppv))
        log("Class TPR: %s" % cls_tpr)
        log("Class PPV: %s" % cls_ppv)
        best_cm.print_normalized_matrix()
        obj_path = os.path.join(meter.save_folder, f"best_cm{phase}_{epoch}")
        best_cm.save(obj_path, best_qwk, epoch_loss, save_stat=True, save_vector=True)
        print()
    """

    lr = opt.param_groups[-1]["lr"]
    # take care of base metrics
    acc, tpr, ppv, f1, cls_tpr, cls_ppv, cls_f1 = get_stats(base_cm)
    log(
        "QWK: %0.4f | ACC: %0.4f | TPR: %0.4f | PPV: %0.4f | F1: %0.4f"
        % (base_qwk, acc, tpr, ppv, f1)
    )
    log(f"Class TPR: {cls_tpr}")
    log(f"Class PPV: {cls_ppv}")
    log(f"Class F1: {cls_f1}")
    base_cm.print_normalized_matrix()
    log(f"lr: {lr}")

    # tensorboard
    logger = tb[phase]

    for cls in cls_tpr.keys():
        logger.log_value("TPR_%s" % cls, float(cls_tpr[cls]), epoch)
        logger.log_value("PPV_%s" % cls, float(cls_ppv[cls]), epoch)
        logger.log_value("F1_%s" % cls, float(cls_f1[cls]), epoch)

    logger.log_value("loss", epoch_loss, epoch)
    if phase == "train":
        logger.log_value("lr", lr, epoch)

    logger.log_value(f"ACC", acc, epoch)
    logger.log_value(f"QWK", base_qwk, epoch)
    logger.log_value(f"TPR", tpr, epoch)
    logger.log_value(f"PPV", ppv, epoch)
    logger.log_value(f"F1", f1, epoch)


    # save pycm confusion
    obj_path = os.path.join(meter.save_folder, f"base_cm{phase}_{epoch}")
    base_cm.save(obj_path, base_qwk, epoch_loss, save_stat=True, save_vector=True)

    return base_qwk


def log_metrics(logger, metrics, epoch, prefix):
    acc, qwk, tpr, ppv = metrics
    logger.log_value(f"{prefix}_ACC", acc, epoch)
    logger.log_value(f"{prefix}_QWK", qwk, epoch)
    logger.log_value(f"{prefix}_TPR", tpr, epoch)
    logger.log_value(f"{prefix}_PPV", ppv, epoch)


def get_stats(cm):
    acc = cm.overall_stat["Overall ACC"]
    tpr = cm.overall_stat["TPR Macro"]  # [7]
    ppv = cm.overall_stat["PPV Macro"]
    f1 = cm.overall_stat["F1 Macro"]
    cls_tpr = cm.class_stat["TPR"]
    cls_ppv = cm.class_stat["PPV"]
    cls_f1 = cm.class_stat["F1"]

    if tpr is "None":
        tpr = 0  # [8]
    if ppv is "None":
        ppv = 0
    if f1 is "None":
        f1 = 0

    cls_tpr = sanitize(cls_tpr)
    cls_ppv = sanitize(cls_ppv)
    cls_f1 = sanitize(cls_f1)

    return acc, tpr, ppv, f1, cls_tpr, cls_ppv, cls_f1


def sanitize(cls_dict):
    for x, y in cls_dict.items():
        try:
            cls_dict[x] = float("%0.4f" % y)
        except Exception as e:  # [8]
            cls_dict[x] = 0.0
    return cls_dict


def check_sanctity(dataloaders):
    phases = dataloaders.keys()
    if len(phases) > 1:
        tnames = dataloaders["train"].dataset.fnames
        vnames = dataloaders["val"].dataset.fnames
        common = [x for x in tnames if x in vnames]
        if len(common):
            print("TRAIN AND VAL SET NOT DISJOINT")
            exit()
    else:
        print("No sanctity check")


def predict(X, coef):
    # [0.15, 2.4, ..] -> [0, 2, ..]
    X_p = np.copy(X)
    for i, pred in enumerate(X_p):
        if pred < coef[0]:
            X_p[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            X_p[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            X_p[i] = 2
        # else:
        #    X_p[i] = 3
        elif pred >= coef[2] and pred < coef[3]:
            X_p[i] = 3
        else:
            X_p[i] = 4
    return X_p.astype("int")

"""Footnotes:

[1]: https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures

[2]: Used in cross-entropy loss, one-hot to single label

[3]: # argmax returns earliest/first index of the maximum value along the given axis
 get_preds ka ye hai ki agar kisi output me zero nahi / sare one hain to 5 nahi to jis index par pehli baar zero aya wahi lena hai, example:
[[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 0, 1, 1, 0], [0, 0, 0, 0, 0]]
-> [4, 1, 0, 0]
baki clip karna hai (0, 4) me, we can get -1 for cases with all zeros.

[4]: get_best_threshold is used in the validation phase, during each phase (train/val) outputs and targets are accumulated. At the end of train phase a threshold of 0.5 is used for
generating the final predictions and henceforth for the computation of different metrics.
Now for the validation phase, best_threshold function is used to compute the optimum threshold so that the qwk is minimum and that threshold is used to compute the metrics.

It can be argued ki why are we using 0.5 for train, then, well we used 0.5 for both train/val so far, so if we are computing this val set best threshold, then not only it can be used to best evaluate the model on val set, it can also be used during the test time prediction as it is being saved with each ckpt.pth

[5]: np.array because it's a list and gets converted to np.array in get_best_threshold function only which is called in val phase and not training phase

[6]: It's important to keep these two in np array, else ConfusionMatrix takes targets as strings. -_-

[7]: macro mean average of all the classes. Micro is batch average or sth.

[8]: sometimes initial values may come as "None" (str)

[9]: I'm using base th for train phase, so base_qwk and best_qwk are same for train phase, helps in comparing the base_qwk and best_qwk of val phase with the train one, didn't find a way to plot base_qwk of train with best and base of val on a single plot.
"""
