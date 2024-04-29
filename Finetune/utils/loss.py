'''Loss function.
'''
import sys
import torch
import torch.nn as nn
# from torch.autograd import Variable
# import numpy as np
# import scipy.ndimage as nd
# from matplotlib import pyplot as plt
# from torch import Tensor, einsum
from monai.losses import DiceCELoss, DiceLoss


def get_loss_function(args):
    '''Get loss function.'''
    if args.loss_name == 'DiceCELoss':
        loss_func = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True,
            smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        ).cuda()
    elif args.loss_name == 'DiceLoss':
        #!! TODO, important to modify
        if args.dataset_name == '10_Decathlon_Task01_BrainTumour':
            loss_func = DiceLoss(smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr, squared_pred=True, to_onehot_y=False, sigmoid=True)
            print('Using multi-labels dice loss....')
        else:
            loss_func = DiceLoss(to_onehot_y=True, softmax=True).cuda()
    elif args.loss_name == 'CE':
        loss_func = nn.CrossEntropyLoss()
    else:
        print('=> unknown loss name', flush=True)
        sys.exit(-1)
    return loss_func


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        """
        Args:
            predict:(B, 1, D, H, W)
            target: (B, 1, D, H, W)
        """
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den
        dice_loss = 1 - dice_score

        dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]

        return dice_loss_avg

# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
#         super(DiceLoss, self).__init__()
#         self.kwargs = kwargs
#         self.weight = weight
#         self.ignore_index = ignore_index
#         self.num_classes = num_classes
#         self.dice = BinaryDiceLoss(**self.kwargs)
#
#     def forward(self, predict, target, name, TEMPLATE):
#         """
#         Args:
#             predict:(B, C, D, H, W)
#             target: (B, C, D, H, W)
#             name: (B, 1)
#             TEMPLATE: dict
#         """
#
#         total_loss = []
#         predict = F.sigmoid(predict)
#
#         total_loss = []
#         B = predict.shape[0]
#
#         for b in range(B):
#             dataset_index = int(name[b][0:2])
#             if dataset_index == 10:
#                 template_key = name[b][0:2] + '_' + name[b][17:19]
#             elif dataset_index == 1:
#                 if int(name[b][-2:]) >= 60:
#                     template_key = '01_2'
#                 else:
#                     template_key = '01'
#             else:
#                 template_key = name[b][0:2]
#             organ_list = TEMPLATE[template_key]
#             for organ in organ_list:
#                 dice_loss = self.dice(predict[b, organ-1], target[b, organ-1])
#                 # print(organ, dice_loss, torch.unique(target[b, organ-1]))
#                 total_loss.append(dice_loss)
#
#         total_loss = torch.stack(total_loss)
#
#         return total_loss.sum()/total_loss.shape[0]



class Multi_BCELoss(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(Multi_BCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target, name, TEMPLATE):
        """
        Args:
            predict:(B, C, D, H, W)
            target: (B, C, D, H, W)
            name: (B, 1)
            TEMPLATE: dict
        """
        assert predict.shape[2:] == target.shape[2:], 'predict & target shape do not match'

        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            dataset_index = int(name[b][0:2])
            if dataset_index == 10:
                template_key = name[b][0:2] + '_' + name[b][17:19]
            elif dataset_index == 1:
                if int(name[b][-2:]) >= 60:
                    template_key = '01_2'
                else:
                    template_key = '01'
            else:
                template_key = name[b][0:2]
            organ_list = TEMPLATE[template_key]
            for organ in organ_list:
                # Compute BCE Loss for each organ.
                ce_loss = self.criterion(predict[b, organ-1], target[b, organ-1])
                total_loss.append(ce_loss)
        total_loss = torch.stack(total_loss)

        return total_loss.sum()/total_loss.shape[0]