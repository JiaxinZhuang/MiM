import sys
import os
from copy import deepcopy
import torch
import monai

# Segmentation models
from monai.networks.nets import UNet, SegResNet
from networks.unetr import UNETR

CODE_PATH = os.environ.get('CODE_PATH')
PRETRAINED_PATH = os.environ.get('PRETRAINED_PATH')
sys.path.append(os.path.join(CODE_PATH, 'MMSMAE_20230904'))
sys.path.append(os.path.join(CODE_PATH, 'MMSMAE_20230904/networks'))
from convvit3d_unetr import ConvViT3dUNETR

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
    elif args.model_name == 'vit_base':
        # from networks.vit import ViT
        # model = ViT(args=args)
        from networks.models_3dvit import vit_base_patch16
        model = vit_base_patch16(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            num_classes=args.out_channels,
            classification=True,
            global_pool=True,
        )
        if args.pretrained_path: # Load pre-trained weights
            model_dict = dict(model.state_dict())
            state_dict = torch.load(args.pretrained_path, map_location='cpu')['model']
            pretrain_dict = {k: v for k, v in state_dict.items() if k in model_dict} # only load the encoder part
            model_dict.update(pretrain_dict)
            model_dict['pos_embed'] = model_dict['pos_embed'][:, 1:, :]
            model_dict = interpolate_pos_embed(args, model, model_dict, new_size=(args.roi_x, args.roi_y, args.roi_z), prefix='')
            out = model.load_state_dict(model_dict)
            print(out)
    elif args.model_name == 'model_genesis':
        from networks import model_genesis_unet3d
        model = model_genesis_unet3d.UNet3D(num_classes=args.out_channels, classification=True)
        model_dict = dict(model.state_dict())
        args.pretrained_path= os.path.join(PRETRAINED_PATH, 'ModelGenesis/Genesis_Chest_CT.pt')
        state_dict = torch.load(args.pretrained_path)['state_dict']
        pretrain_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.replace('module.', '') in model_dict and not k.startswith('module.out_tr')}
        model_dict.update(pretrain_dict)
        out = model.load_state_dict(pretrain_dict, strict=False)
        print(out)
    # elif args.model_name == 'GVSL':
    #     model = UNet_GVSL.UNet3D_GVSL(n_classes=args.out_channels)
    #     ckpt_path = '/jhcnas1/jiaxin/ckpts/pretrained_weights/GVSL_epoch_1000.pth'
    #     out = model.unet_pre.load_state_dict(torch.load(ckpt_path))
    #     print(out)
    elif args.model_name == 'GL-MAE':
        from networks.models_3dvit import vit_base_patch16
        model = vit_base_patch16(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            num_classes=args.out_channels,
            classification=True,
            global_pool=True,
        )
        if PRETRAINED_PATH: # Load pre-trained weights
            if args.pretrained_path is None:
                args.pretrained_path = os.path.join(PRETRAINED_PATH, 'DINO/checkpoint-50.pth')
            model_dict = dict(model.state_dict())
            state_dict = torch.load(args.pretrained_path, map_location='cpu')['student']
            # print('--', state_dict.keys())
            # print('**', model_dict.keys())
            pretrain_dict = {k.replace('module.backbone.', ''): v for k, v in state_dict.items() if k.replace('module.backbone.', '') in model_dict} # only load the encoder part
            model_dict.update(pretrain_dict)
            model_dict['pos_embed'] = model_dict['pos_embed'][:, 1:, :]
            model_dict = interpolate_pos_embed(args, model, model_dict, new_size=(args.roi_x, args.roi_y, args.roi_z), prefix='')
            out = model.load_state_dict(model_dict)
            print(out)
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
    elif args.model_name == 'MiM':
        from models_convvit3d_cls import convvit3d_base_patch16_CC_CCII
        model = convvit3d_base_patch16_CC_CCII(num_classes=args.out_channels, global_pool=True, args=args)
        if args.pretrained_path:
            model_dict = model.state_dict()
            state_dict = torch.load(args.pretrained_path, map_location='cpu')['model']
            pretrain_dict = {k: v for k, v in state_dict.items() if k in model_dict} # only load the encoder part
            model_dict.update(pretrain_dict)
            model_dict = interpolate_pos_embed(args, model, model_dict, new_size=(args.roi_x, args.roi_y, args.roi_z), prefix='')
            out = model.load_state_dict(model_dict)
            print(out)
    else:
        raise NotImplementedError
    return model


def get_seg_model(args=None):
    '''Get segmentation model.'''
    if args.model_name in ['vit_base', 'GL-MAE', 'swin_unetr']:
        num_heads = 12
        num_layer = 12
        hidden_size = 768
        if args.feature_size is None:
            feature_size = 48
        else:
            feature_size = args.feature_size
    elif args.model_name == 'vit_small':
        num_heads = 6
        num_layer = 12
        hidden_size = 384
        feature_size = 24
    elif args.model_name in ['vit_tiny', 'convit3d_tiny', 'swin_unetr_tiny']:
        num_heads = 3
        num_layer = 12
        hidden_size = 192
        feature_size = 12
    elif args.model_name == 'vit_large':
        num_heads = 16
        num_layer = 24
        hidden_size = 1152
        feature_size = 96
    elif args.model_name == 'vit_huge':
        num_heads = 16
        num_layer = 32
        hidden_size = 1344
        feature_size = 192
    elif args.model_name in ['unet',
                             'segresnet', 'MoCoV2_segresnet',
                             'convit3d',
                             'model_genesis', 'GVSL',
                             'MiT',
                             'jigsaw_swin', 'rubik_swin', 'positionLabel_swin',
                             'HPM_mae_vit_base_patch16', 'localMIM_vit_base_patch16', 'Adam', 'simMIM_swin',
                             'PCRL', 'PCRLv2']:
        feature_size = None
    else:
        sys.exit(-1)

    if feature_size is not None and args.rank == 0:
        args.feature_size = feature_size
        print('Force feature size to: ', args.feature_size)

    if args.model_name == 'unet':
        model = UNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif args.model_name in ['segresnet', 'MoCoV2_segresnet']:
        model = SegResNet(in_channels=args.in_channels, out_channels=args.out_channels)
        if args.pretrained_path:
            model_dict = dict(model.state_dict())
            state_dict = torch.load(args.pretrained_path, map_location='cpu')['model']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith("encoder_q") and not k.startswith("encoder_q.fc"):
                    # remove prefix
                    state_dict[k[len("encoder_q.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            pretrain_dict = {k: v for k, v in state_dict.items() if k in model_dict} # only load the encoder part
            not_pretrain_dict = set(state_dict.keys()) - set(pretrain_dict.keys())
            model_dict.update(pretrain_dict)
            out = model.load_state_dict(model_dict)

            print(model_dict.keys())
            print(f'not loading keys {not_pretrain_dict}')
            print(out)
    elif args.model_name.startswith('swin_unetr'):
        from monai.networks.nets import SwinUNETR
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
            except:
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
        if args.freeze_backbone:
            # Freeze the parameters of the backbone
            for param in model.swinViT.parameters():
                param.requires_grad = False
            # Verify that the backbone parameters are frozen
            for name, param in model.named_parameters():
                print(f'Freezing {name} {param.requires_grad}')
    elif args.model_name in ['convit3d', 'convit3d_tiny']:
        model = ConvViT3dUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                               in_channels=args.in_channels,
                               out_channels=args.out_channels,
                               feature_size=args.feature_size,
                               args=args,
                               )
        if args.freeze_backbone:
            # Freeze the parameters of the backbone
            for param in model.backbone.parameters():
                param.requires_grad = False
            # Verify that the backbone parameters are frozen
            for name, param in model.named_parameters():
                print(f'Freezing {name} {param.requires_grad}')
    elif args.model_name == 'GL-MAE':
        mlp_dim = hidden_size * 4
        args_copy = deepcopy(args)
        args_copy.model_name = 'vit_base'
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layer=num_layer,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.0,
            args=args_copy
        )
    elif args.model_name in ['vit_tiny', 'vit_small', 'vit_large', 'vit_huge', 'vit_base']:
        mlp_dim = hidden_size * 4
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layer=num_layer,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.0,
            args=args
        )
    elif args.model_name == 'model_genesis':
        from networks import model_genesis_unet3d
        model = model_genesis_unet3d.UNet3D(in_channels=args.in_channels, num_classes=args.out_channels)
        model_dict = dict(model.state_dict())
        args.pretrained_path= os.path.join(PRETRAINED_PATH, 'ModelGenesis/Genesis_Chest_CT.pt')
        state_dict = torch.load(args.pretrained_path, map_location='cpu')['state_dict']
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        filter_keys = ['out_tr.final_conv.weight', 'out_tr.final_conv.bias']
        pretrain_dict = {k: v for k, v in state_dict.items() if k in model_dict and k not in filter_keys}
        not_pretrain_dict = set(state_dict.keys()) - set(pretrain_dict.keys())
        checkpoint_weight = pretrain_dict['down_tr64.ops.0.conv1.weight']
        repeated_weight = checkpoint_weight.repeat(1, args.in_channels, 1, 1, 1)
        pretrain_dict['down_tr64.ops.0.conv1.weight'] = repeated_weight
        model_dict.update(pretrain_dict)
        out = model.load_state_dict(model_dict)

        print(model_dict.keys())
        print(f'not loading keys {not_pretrain_dict}')
        print(out)
    elif args.model_name == 'PCRL':
        from networks.pcrl_model_3d import PCRLModel3d
        model = PCRLModel3d(num_classes=args.out_channels, student=True)
    elif args.model_name == 'PCRLv2':
        from networks.pcrlv2_model_3d import SegmentationModel
        model = SegmentationModel(num_classes=args.out_channels)
        model_dict = dict(model.state_dict())
        # '/jhcnas1/jiaxin/ckpts/pretrained_weights/simance_multi_crop_luna_pretask_1.0_240.pt'
        if PRETRAINED_PATH and not args.pretrained_path:
            args.pretrained_path= os.path.join(PRETRAINED_PATH, 'pcrlv2/simance_multi_crop_luna_pretask_1.0_240.pt')

        state_dict = torch.load(args.pretrained_path)['state_dict']
        # pretrain_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'down_tr' in k} # only load the encoder part
        filter_keys = ['out_tr.final_conv.weight', 'out_tr.final_conv.bias']
        pretrain_dict = {k: v for k, v in state_dict.items() if k in model_dict and k not in filter_keys} # only load the encoder part
        not_pretrain_dict = set(state_dict.keys()) - set(pretrain_dict.keys())
        model_dict.update(pretrain_dict)
        out = model.load_state_dict(model_dict)

        print(model_dict.keys())
        print(f'not loading keys {not_pretrain_dict}')
        print(out)
    elif args.model_name == 'GVSL':
        from networks.UNet_GVSL import UNet3D_GVSL
        model = UNet3D_GVSL(in_channels=args.in_channels, n_classes=args.out_channels)
        model_dict = dict(model.state_dict())
        #pretrained_path = '/jhcnas1/jiaxin/ckpts/pretrained_weights/GVSL_epoch_1000.pth'
        if PRETRAINED_PATH and not args.pretrained_path:
            args.pretrained_path = os.path.join(PRETRAINED_PATH, 'GVSL/GVSL_epoch_1000.pth')
        state_dict = torch.load(args.pretrained_path, map_location='cpu')
        filter_keys = []
        pretrain_dict = {k: v for k, v in state_dict.items() if k in model_dict and k not in filter_keys}
        not_pretrain_dict = set(state_dict.keys()) - set(pretrain_dict.keys())
        out = model.load_state_dict(model_dict)

        print(model_dict.keys())
        print(f'not loading keys {not_pretrain_dict}')
        print(out)
    elif args.model_name == 'MiT':
        from networks.MiT import MiTnet
        # args.pretrained_path = '/jhcnas1/jiaxin/ckpts/pretrained_weights/UniMiss_small.pth'
        if PRETRAINED_PATH and not args.pretrained_path:
            args.pretrained_path = os.path.join(PRETRAINED_PATH, 'unimiss/UniMiss_small.pth')
        print(f'Loading pretrained weights from {args.pretrained_path}')
        model = MiTnet(norm_cfg='BN', activation_cfg='ReLU', weight_std=False, img_size=(args.roi_x, args.roi_y, args.roi_z),
                    num_classes=args.out_channels, in_chans=args.in_channels,
                    pretrain=True if args.pretrained_path else False, pretrain_path=args.pretrained_path, deep_supervision=False)
    elif args.model_name in ['jigsaw_swin', 'rubik_swin', 'positionLabel_swin']:
        from monai.networks.nets import SwinUNETR
        args.feature_size = 48
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=1,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=True if args.pretrained_path is not None else False,
        )
        if args.model_name == 'rubik_swin' and not args.pretrained_path:
            args.pretrained_path= os.path.join(PRETRAINED_PATH, 'rubik/rubik++_final_model.pth')
        elif args.model_name == 'jigsaw_swin' and not args.pretrained_path:
            args.pretrained_path= os.path.join(PRETRAINED_PATH, 'jigsaw/jigsaw_final_model.pth')

        print(f'Loading pretrained weights from {args.pretrained_path}')
        state_dict = torch.load(args.pretrained_path)
        # ['state_dict']
        if args.pretrained_path:
            model_dict = dict(model.state_dict())
            state_dict = torch.load(args.pretrained_path, map_location='cpu')
            filter_keys = []
            pretrain_dict = {k: v for k, v in state_dict.items() if k in model_dict and k not in filter_keys}
            not_pretrain_dict = set(state_dict.keys()) - set(pretrain_dict.keys())
            model_dict.update(pretrain_dict)
            out = model.load_state_dict(model_dict)

            print(model_dict.keys())
            print(f'not loading keys {not_pretrain_dict}')
            print(out)
    elif args.model_name == 'HPM_mae_vit_base_patch16':
        args.model_name = 'vit_base'
        num_heads = 12
        num_layer = 12
        hidden_size = 768
        feature_size = 48
        mlp_dim = hidden_size * 4
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layer=num_layer,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.0,
            args=args
        )
        if PRETRAINED_PATH or args.pretrained_path:
            if args.pretrained_path is None:
                args.pretrained_path= os.path.join(PRETRAINED_PATH, 'HPM/checkpoint-final.pth')
            model_dict = dict(model.state_dict())
            state_dict = torch.load(args.pretrained_path, map_location='cpu')['model']
            filter_keys = []
            # special process
            pretrain_dict = {}
            for k, v in state_dict.items():
                key = 'vit.' + k
                if key == 'vit.pos_embed':
                    value = v[:, 1:,]
                else:
                    value = v
                if key in model_dict and key not in filter_keys:
                    pretrain_dict[key] = value
            not_pretrain_dict = set(state_dict.keys()) - set(pretrain_dict.keys())
            model_dict.update(pretrain_dict)
            out = model.load_state_dict(model_dict)

            print(f'Successfully loaded keys: {pretrain_dict.keys()}')
            print(f'not loading keys {not_pretrain_dict}')
    elif args.model_name == 'localMIM_vit_base_patch16':
        args.model_name = 'vit_base'
        num_heads = 12
        num_layer = 12
        hidden_size = 768
        feature_size = 48
        mlp_dim = hidden_size * 4
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layer=num_layer,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.0,
            args=args
        )
        if PRETRAINED_PATH or args.pretrained_path:
            if args.pretrained_path is None:
                args.pretrained_path= os.path.join(PRETRAINED_PATH, 'localMIM/checkpoint-799.pth')
            model_dict = dict(model.state_dict())
            state_dict = torch.load(args.pretrained_path, map_location='cpu')['model']
            filter_keys = []
            # special process
            pretrain_dict = {}
            for k, v in state_dict.items():
                key = 'vit.' + k
                if key == 'vit.pos_embed':
                    value = v[:, 1:,]
                else:
                    value = v
                if key in model_dict and key not in filter_keys:
                    pretrain_dict[key] = value
            not_pretrain_dict = set(state_dict.keys()) - set(pretrain_dict.keys())
            model_dict.update(pretrain_dict)
            out = model.load_state_dict(model_dict)

            print(f'Successfully loaded keys: {pretrain_dict.keys()}')
            print(f'not loading keys {not_pretrain_dict}')
    elif args.model_name in ['Adam']:
        model = SegResNet(in_channels=args.in_channels, out_channels=args.out_channels)
        if PRETRAINED_PATH and not args.pretrained_path:
            args.pretrained_path= os.path.join(PRETRAINED_PATH, 'Adam/Pretraining_1k_MoCov2_n4_resnet_240127/checkpoint-249.pth')
        model_dict = dict(model.state_dict())
        state_dict = torch.load(args.pretrained_path, map_location='cpu')['model']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("encoder_q") and not k.startswith("encoder_q.fc"):
                # remove prefix
                state_dict[k[len("encoder_q.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        pretrain_dict = {k: v for k, v in state_dict.items() if k in model_dict} # only load the encoder part
        not_pretrain_dict = set(state_dict.keys()) - set(pretrain_dict.keys())
        model_dict.update(pretrain_dict)
        out = model.load_state_dict(model_dict)

        print(f'Successfully loaded keys: {pretrain_dict.keys()}')
        print(f'not loading keys {not_pretrain_dict}')
            # print(model_dict.keys())
            # print(f'not loading keys {not_pretrain_dict.keys()}')
            # print(out)
    elif args.model_name in ['simMIM_swin']:
        from networks.simMIM.swin_unetr_og import SwinUNETR
        # from networks.simMIM.swin_unetr import SwinUNETR
        model = SwinUNETR(
           img_size=(args.roi_x, args.roi_y, args.roi_z),
           in_channels=args.in_channels,
           out_channels=args.out_channels,
           feature_size=args.feature_size,
           drop_rate=0.0,
           attn_drop_rate=0.0,
           dropout_path_rate=args.dropout_path_rate,
           use_checkpoint=args.use_checkpoint,
        )
        # if load pretrained ckpt
        if PRETRAINED_PATH or args.pretrained_path:
            if not args.pretrained_path:
                args.pretrained_path= os.path.join(PRETRAINED_PATH, 'simMIM/Pretraining_1k_simMIM_swin_240127/checkpoint-final.pth')
            weight = torch.load(args.pretrained_path)
            model.load_from(weights=weight, finetune_choice='encoder')
            print(f"Successfully loaded keys: {weight.keys()}")
    elif args.model_name.startswith('swin_unetr'):
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=True if args.pretrained_path is not None else False,
        )
    else:
        print('Require valid model name')
        raise NotImplementedError
    return model
