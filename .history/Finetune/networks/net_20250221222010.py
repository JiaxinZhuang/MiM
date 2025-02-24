"""Jiaxin ZHUANG.
Modified on April 29th, 2024.
"""

import sys
import os
import torch
import monai

CODE_PATH = os.environ.get('CODE_PATH')
PRETRAINED_PATH = os.environ.get('PRETRAINED_PATH')

from utils.downstream_utils import interpolate_pos_embed


def get_model(args):
    """Get model.
    """
    if args.task == 'cls':
        model = get_cls_model(args)
    elif args.task == 'seg':
        model = get_seg_model(args)
    else:
        raise NotImplementedError
    return model


def get_cls_model(args=None):
    '''Get classification model.'''
    if args.model_name.startswith('resnet'):
        model = monai.networks.nets.resnet.__dict__[args.model_name](n_input_channels=args.in_channels,
                                                                     num_classes=args.out_channels)
    elif args.model_name == 'DenseNet':
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=args.out_channels)
    elif args.model_name == 'SENet154':
        model = monai.networks.nets.SENet154(spatial_dims=3, in_channels=1, num_classes=args.out_channels)
        raise NotImplementedError #!! TODO
    elif args.model_name == 'vit_base':
        raise NotImplementedError #!! TODO
    elif args.model_name == 'model_genesis':
        raise NotImplementedError #!! TODO
    elif args.model_name == 'GL-MAE':
        raise NotImplementedError #!! TODO
    elif args.model_name == 'swin':
        from networks.swin import SwinTransformer
        model = SwinTransformer(in_chans=args.in_channels, embed_dim=args.feature_size,
                                window_size=[7, 7, 7],
                                patch_size=[2, 2, 2],
                                depths=[2, 2, 2, 2],
                                num_heads=[3, 6, 8, 24],
                                num_classes=args.out_channels,)
        print(model)
        if args.pretrained_path:
            model_dict = dict(model.state_dict())
            state_dict = torch.load(args.pretrained_path, map_location='cpu')['state_dict']
            pretrain_dict = {k: v for k, v in state_dict.items() if k in model_dict} # only load the encoder part
            model_dict.update(pretrain_dict)
            out = model.load_state_dict(model_dict)
            print(out)
        raise NotImplementedError #!! TODO
    elif args.model_name == 'MiM':
        raise NotImplementedError #!! TODO
    else:
        raise NotImplementedError
    return model


def get_seg_model(args=None):
    '''Get segmentation model.'''
    if args.model_name == 'unet':
        raise NotImplementedError #!! TODO
    elif args.model_name in ['segresnet', 'MoCoV2_segresnet']:
        raise NotImplementedError #!! TODO
    elif args.model_name.startswith('swin_unetr'):
        from monai.networks.nets import SwinUNETR
        args.feature_size = 48 if args.feature_size is None else args.feature_size
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=True if args.pretrained_path is not None else False,
        )
        if args.pretrained_path is not None:
            model_dict = dict(model.state_dict())
            try:
                state_dict = torch.load(args.pretrained_path, map_location='cpu')['state_dict']
            except Exception as e:
                state_dict = torch.load(args.pretrained_path, map_location='cpu')['model']
            state_dict = {k.replace('fc', 'linear'): v for k, v in state_dict.items()}
            pretrain_dict = {k.replace('module.', 'swinViT.'): v for k, v in state_dict.items() if k.replace('module.', 'swinViT.') in model_dict} # only load the encoder part
            not_pretrain_dict = {k: v for k, v in state_dict.items() if k.replace('module.', 'swinViT.') not in model_dict} # only load the encoder part
            checkpoint_weight = pretrain_dict['swinViT.patch_embed.proj.weight']
            repeated_weight = checkpoint_weight.repeat(1, args.in_channels, 1, 1, 1)
            pretrain_dict['swinViT.patch_embed.proj.weight'] = repeated_weight
            model_dict.update(pretrain_dict)
            out = model.load_state_dict(model_dict)

            print(model_dict.keys())
            print(f'not loading keys {not_pretrain_dict.keys()}')
            print(out)
    elif args.model_name == 'GL-MAE':
        raise NotImplementedError #!! TODO
    elif args.model_name in ['vit_tiny', 'vit_small', 'vit_large', 'vit_huge', 'vit_base']:
        raise NotImplementedError #!! TODO
    elif args.model_name == 'model_genesis':
        raise NotImplementedError #!! TODO
    elif args.model_name == 'PCRL':
        raise NotImplementedError #!! TODO
    elif args.model_name == 'PCRLv2':
        raise NotImplementedError #!! TODO
    elif args.model_name == 'GVSL':
        raise NotImplementedError #!! TODO
    elif args.model_name == 'MiT':
        raise NotImplementedError #!! TODO
    elif args.model_name in ['jigsaw_swin', 'rubik_swin', 'positionLabel_swin']:
        raise NotImplementedError #!! TODO
    elif args.model_name == 'HPM_mae_vit_base_patch16':
        raise NotImplementedError #!! TODO
    elif args.model_name == 'localMIM_vit_base_patch16':
        raise NotImplementedError #!! TODO
    elif args.model_name in ['Adam']:
        raise NotImplementedError #!! TODO
    elif args.model_name in ['simMIM_swin']:
        raise NotImplementedError #!! TODO
    else:
        print('Require valid model name')
        raise NotImplementedError
    return model
