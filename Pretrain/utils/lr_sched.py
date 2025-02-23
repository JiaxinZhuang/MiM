# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_sin_weight_higher(args=None, epoch=None):
    '''Get the weight for loss iterm from 0 to 1'''
    epoch, max_epochs = torch.tensor(epoch), torch.tensor(args.epochs)
    sin_value = torch.sin(torch.pi * epoch / (2 * max_epochs))
    return sin_value

def get_cos_weight_lower(args=None, epoch=None):
    '''Get the weight for loss iterm from 2 to 1'''
    epoch, max_epochs = torch.tensor(epoch), torch.tensor(args.epochs)
    cos_value = torch.cos(torch.pi * epoch / (2 * max_epochs)) + torch.tensor(1.0)
    return cos_value