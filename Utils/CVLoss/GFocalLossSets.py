#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : elfin
# @Time     : 2021/10/12 11:37
# @File     : GFocalLossSets.py
# @Project  : Loss
import numbers
import torch.nn as nn


class QFocalLoss(nn.Module):
    r""" Quality focal loss
    Wraps Quality focal loss around existing loss_fcn(),
    i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    mathematical formula:

    .. math::
        -\left | y-\sigma  \right |^{\beta }\left ( \left ( 1-\alpha \right ) \left ( 1-y \right )
        \log\left ( 1-\sigma  \right ) + \alpha \times y \log\left ( \sigma  \right ) \right )
    Where :math:`y` is the label and :math:`\sigma` is the predicted value after Logits.

    :Reference
        https://github.com/ultralytics/yolov5/blob/master/utils/loss.py (line:65)

    :Args
        - loss_fcn: nn.BCEWithLogitsLoss object;
        - gamma: gamma controls the down-weighting rate smoothly for modulating factor;
        - alpha: positive and negative sample balance parameters.

    :Shape
    """

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class DFocalLoss(nn.Module):
    r"""
    """
    pass
