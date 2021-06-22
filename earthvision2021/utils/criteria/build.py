import torch
import torch.nn as nn

from utils.criteria.jaccard_loss import JaccardLoss
from utils.criteria.dice_loss import DiceLoss


def build(loss_name, loss_args, **kwargs):
    if loss_args is not None and 'weight' in loss_args:
        _weight = loss_args['weight']
        loss_args['weight'] = torch.tensor(_weight) \
                if not isinstance(_weight, torch.Tensor) else _weight

    loss_dict = {
            'ce': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
            'mae': nn.L1Loss,
            'jaccard': JaccardLoss,
            'dice': DiceLoss,
    }
    return loss_dict[loss_name.lower()](**(loss_args or {}))

