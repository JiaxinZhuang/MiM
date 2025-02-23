"""Jiaxin ZHUANG.
Modified Aug 25, 2023.
"""

import argparse


def get_args():
    '''Get default arguments.'''
    parser = argparse.ArgumentParser('Pre-training', add_help=False)
    parser.add_argument("--tags", nargs="+", type=int, help="tags of training")

    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model_name', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset_split', default='1k', type=str,
                        choices=['1k', '+2k', '+4k', '+10k', '+20k', '+100k', '+110k'],
                        help='dataset split.')
    parser.add_argument('--json_dir', default='/data/CT/data/jsons', type=str,
                        help='json directory.')
    parser.add_argument('--yaml_path', default='./configs/datasets.yaml', type=str,
                        help='dataset path')
    parser.add_argument('--data_path', default='/dev/shm/data', type=str,
                        help='dataset path')
    parser.add_argument('--cache_dir', default='/data/tmp/', type=str,
                        help='cache directory to store dataset.')
    # parser.add_argument('--use_ssd', action='store_true',
                        # help='whether to use dev/shm to store some data\
                            # set for training')

#     parser.add_argument('--output_dir', default='./output_dir',
                        # help='path where to save, empty for no saving')
    parser.add_argument('--logdir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--a_min', type=float, default=-1000,
                        help='min for threshold')
    parser.add_argument('--a_max', type=float, default=1000,
                        help='max for threshold')
    parser.add_argument('--b_min', type=float, default=0.0,
                        help='min for threshold')
    parser.add_argument('--b_max', type=float, default=1.0,
                        help='max for threshold')

    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
    parser.add_argument("--persistent_dataset", action="store_true", help="use monai persistent Dataset")
    parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")

    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")

    parser.add_argument("--eval", action='store_true', help="eval mode for transform")

    parser.add_argument("--save_fq", default=5, type=int, help="save frequency")

    parser.add_argument('--dataset_loader', default='get_loader', type=str,
                        # choices=['v3', 'v2', 'v1', 'mmsmae', 'MoCoV2', 'Adam', 'HPM', 'simMIM'],
                        help='dataloader.')

    parser.add_argument("--decoder_embed_dim", default=576, type=int, help="decoder_embed_dim")

    parser.add_argument('--normalize', action='store_true', help='normalize the data')

    parser.add_argument("--sr_ratio", default=1, type=int, help="multi scale token")

    # MMSMAE.
    parser.add_argument("--crop_x", default=96*2, type=int, help="roi size in x direction for the overall input")
    parser.add_argument("--crop_y", default=96*2, type=int, help="roi size in y direction for the overall input")
    parser.add_argument("--crop_z", default=96*2, type=int, help="roi size in z direction for the overall input")
    parser.add_argument("--up_roi_x", default=96*1, type=int, help="roi size in x direction for the overall input")
    parser.add_argument("--up_roi_y", default=96*1, type=int, help="roi size in y direction for the overall input")
    parser.add_argument("--up_roi_z", default=96*1, type=int, help="roi size in z direction for the overall input")
    parser.add_argument("--down_roi_x", default=96, type=int, help="roi size in x direction for the overall input")
    parser.add_argument("--down_roi_y", default=96, type=int, help="roi size in y direction for the overall input")
    parser.add_argument("--down_roi_z", default=96, type=int, help="roi size in z direction for the overall input")
    parser.add_argument('--mask_ratio_up', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_ratio_mid', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--mask_ratio_down', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--sample_usual', default=4, type=int, help='Sample volumes for usual stage.')
    parser.add_argument('--sample_down',  default=4, type=int, help='Sample volumes for down stage.')

    parser.add_argument('--not_use_attn', action='store_true',
                        help='whether to use attention')
                        #, dest='use_attn',
    # parser.set_defaults(use_attn=True)

    # MoCo v2.
    parser.add_argument("--moco-dim", default=128, type=int, help="feature dimension (default: 128)")
    parser.add_argument("--moco-k", default=65536, type=int, help="queue size; number of negative keys (default: 65536)",)
    parser.add_argument("--moco-m", default=0.999, type=float, help="moco momentum of updating key encoder (default: 0.999)",)
    parser.add_argument("--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)")
    # options for moco v2
    parser.add_argument("--mlp", action="store_true", help="use mlp head")

    # MiM
    parser.add_argument("--atten_weight_uu", default=1e-3, type=float, help="attention weight for up and usual")
    parser.add_argument("--atten_weight_ud", default=1e-3, type=float, help="attention weight for down and usual")
    parser.add_argument("--reconstruct_weight_up", default=1, type=float,    help="weight for up")
    parser.add_argument("--reconstruct_weight_usual", default=1, type=float, help="weight for usual")
    parser.add_argument("--reconstruct_weight_down", default=1, type=float,  help="weight for down")

    parser.add_argument('--mode', default='concurrent', type=str, choices=['concurrent', 'coarse_to_fine', 'fine_to_coarse'], help='weight for loss.')
    parser.add_argument('--attn_loss_name', default='CE', type=str, choices=['CE', 'byol'], help='loss function.')

    # Adam
    parser.add_argument('--sim_threshold', default=0.8, type=float,help='similarity threshold for purposive pruner')
    parser.add_argument('--granularity', default=0, type=int,help=' data granularity level')
    parser.add_argument('--encoder_weights', default=None, type=str,help='encoder pre-trained weights')
    parser.add_argument('--crop_scale_min', default=0.2, type=float,help='min scale for random crop augmentation')

    # HPM
    parser.add_argument('--vis_mask_ratio', default=0.0, type=float,
                        help='Secondary masking ratio (mask percentage of visible patches, secondary masking phase).')
    parser.add_argument('--learning_loss', action='store_true', help='Learn to predict loss for each patch.')
    parser.set_defaults(learning_loss=True)
    parser.add_argument('--learn_feature_loss', default='none', type=str, help='Use MSE loss for features as target.')
    parser.add_argument('--relative', action='store_true', help='Use relative learning loss or not.')
    parser.add_argument('--token_size', default=int(96 / 16), type=int, help='number of patch (in one dimension), usually input_size//16')  # for mask generator

    # simMIM
    parser.add_argument("--num_classes", default=0, type=int, help="number of input channels")
    parser.add_argument("--window_size", default=(7, 7, 7), type=tuple, help="window size")
    parser.add_argument("--patch_size", default=(2, 2, 2), type=tuple, help="window size")
    parser.add_argument("--mask_patch_size", default=16, type=int, help="window size")
    parser.add_argument("--num_heads", default=[3, 6, 12, 24], type=list, help="number of heads")
    parser.add_argument("--depths", default=[2, 2, 2, 2], type=list, help="number of depths")
    parser.add_argument("--embed_dim", default=48, type=int, help="embedding dimention")
    parser.add_argument("--mlp_ratio", default=4.0, type=float, help="MLP ratio")
    parser.add_argument("--drop_rate", default=0.0, type=float, help="drop rate")
    parser.add_argument("--attn_drop_rate", default=0.0, type=float, help="attention drop rate")
    parser.add_argument("--drop_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--layer_decay", default=1.0, type=float, help="layer decay")
    parser.add_argument("--decoder", type=str, default="upsample", help="decoder type")
    parser.add_argument("--loss_type", type=str, default="mask_only", help="decoder type")
    parser.add_argument("--img_size", default=96, type=int, help="image size")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument(
        "--use_grad_checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
    )

    # SwinUNETR
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")

    parser.add_argument('--test_one_epoch', action='store_true',
                        help='test for one epoch')

    return parser