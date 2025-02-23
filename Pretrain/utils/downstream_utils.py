"""Jiaxin ZHUANG.
Modified on Jun 26, 2023.
"""

import sys, os
import torch
from collections import OrderedDict

from utils.misc import print_with_timestamp


def load_ckpt(args, model):
    """Load checkpoint from pretrained ViT for UNETR.
    """
    if args.pretrained_path:
        if os.path.isfile(args.pretrained_path):
            # print("=> loading checkpoint '{}'".format(args.pretrained_path))
            checkpoint = torch.load(args.pretrained_path, map_location='cpu')
            ckpt = {}
            # Modify the key names in checkpoint.
            for key, value in checkpoint['model'].items():
                new_key = 'vit.{}'.format(key)
                if new_key in ['vit.pos_embed']:
                    # Remove the class token.
                    ckpt[new_key] = value[:, 1:, :]
                else:
                    ckpt[new_key] = value
            out = model.load_state_dict(ckpt, strict=False)
            if args.rank == 0:
                print(out)
                print("=> loaded checkpoint '{}' (epoch {})".format(args.pretrained_path, checkpoint['epoch']))
            del ckpt, checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained_path))
            sys.exit(-1)
    else:
        print("=> no checkpoint")
        # sys.exit(-1)
    return model


def resume_ckpt(args, model, optimizer, scheduler):
    """Resume checkpoint from previous training.
    """
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        out = model.load_state_dict(ckpt["state_dict"], strict=True)
        print_with_timestamp('Load model ckpt {}'.format(out))

        # new_state_dict = OrderedDict()
        # for k, v in checkpoint["state_dict"].items():
            # new_state_dict[k.replace("backbone.", "")] = v
        # model.load_state_dict(new_state_dict, strict=False)
        if 'optimizer' in ckpt:
            out = optimizer.load_state_dict(ckpt["optimizer"])
            print_with_timestamp('Load optimizer: {}'.format(out))
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            print_with_timestamp('Load schduler: {}'.format(out))
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]
            print_with_timestamp('Load start epoch: {}'.format(start_epoch))
        if "best_acc" in ckpt:
            best_acc = ckpt["best_acc"]
            print_with_timestamp('Load best acc: {}'.format(best_acc))
        print_with_timestamp("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.resume, 
                                                                                        start_epoch, 
                                                                                        best_acc))
        # print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))
    args.start_epoch = start_epoch
    args.best_acc = best_acc
    return model, optimizer, scheduler