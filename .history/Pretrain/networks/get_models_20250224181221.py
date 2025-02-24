

import torch
from timm.utils import ModelEma


def build_models(args):
    '''Build model and ema model.'''
    device = torch.device(args.device)
    model_ema = None
    if args.dataset_loader == 'MiM' and args.dataset_loader == 'v1':
        import models_convmae3d_v4
        model = models_convmae3d_v4.__dict__[args.model_name](norm_pix_loss=args.norm_pix_loss, args=args)
    elif args.model_name in ['MoCoV2']:
        from networks.segrenset_encoder import SegResNet as encoder
        import networks.MoCoV2.builder
        model = networks.MoCoV2.builder.MoCo(
            encoder, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp,)
    elif args.model_name == 'Adam':
        from networks.segrenset_encoder import SegResNet as encoder
        import networks.Adam.builder
        model = networks.Adam.builder.MoCo(
            encoder, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, args.encoder_weights)
    elif args.model_name in ['mae_vit_base_patch16']:
        import models_mae
        model = models_mae.__dict__[args.model_name](norm_pix_loss=args.norm_pix_loss)
    elif args.model_name in ['HPM_mae_vit_base_patch16']:
        from networks.HPM import models_mae_learn_loss
        model_name = args.model_name.split('_', maxsplit=1)[-1]
        model = models_mae_learn_loss.__dict__[model_name](norm_pix_loss=args.norm_pix_loss,
                                                                vis_mask_ratio=args.vis_mask_ratio)
        model.to(device)
        # define ema model
        if args.learning_loss or args.learn_feature_loss == 'ema':
            # use momentum encoder for BYOL
            model_ema = ModelEma(model, decay=0.999, device=device, resume='')
    elif args.model_name in ['localMIM_vit_base_patch16']:
        from networks.localMIM import localMIM_models_mim
        model_name = args.model_name.split('_', maxsplit=1)[-1]
        model = localMIM_models_mim.__dict__[model_name](norm_pix_loss=args.norm_pix_loss)
    elif args.model_name in ['simMIM_swin', 'simMIM_vit_base']:
        from networks.simMIM.build import build_model
        args.model_type = args.model_name.split('_')[-1]
        model = build_model(args, is_pretrain=True)
    elif args.model_name in ['SwinUNETR']:
        from networks.SwinUNETR.ssl_head import SSLHead
        model = SSLHead(args)
    elif args.model_name in ['GVSL']:
        from networks.GVSL.GVSL import GVSL
        model = GVSL()
    else:
        raise NotImplementedError
    return model, model_ema