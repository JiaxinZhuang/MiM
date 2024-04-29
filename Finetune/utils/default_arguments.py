"""Jiaxin ZHUANG.
Modified Apirl 30, 2024.
"""

import argparse


def get_args():
    '''Get arguments.'''
    parser = argparse.ArgumentParser(description="Segmentation/Classification for 3D medical images via end-to-end training")
    parser.add_argument("--logdir", default=None, type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--data_dir", default=None, type=str, help="dataset directory")
    parser.add_argument("--json_list", default=None, type=str, help="dataset json file")
    parser.add_argument("--pretrained_path", default=None, type=str, help="pretrained checkpoint path")
    parser.add_argument("--resume", default=None, type=str, help="resume checkpoint path")
    parser.add_argument("--config_path", default="./configs/downstream_configs.yaml",
                        type=str, help="dataset config path.")

    parser.add_argument("--max_epochs", default=None, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=None, type=int, help="number of batch size")
    parser.add_argument("--accum_iter", default=None, type=int, help="accumulate iteration for update")
    parser.add_argument("--infer_sw_batch_size", default=2, type=int, help="number of batch size when running inference slicing windows.")
    parser.add_argument("--num_workers", default=None, type=int, help="number workers for dataloader.")

    parser.add_argument("--optim_lr", default=None, type=float, help="optimization learning rate")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--dist-url", default=None, type=str, help="distributed url, i.e., tcp://127.0.0.1:23456")
    parser.add_argument("--feature_size", default=None, type=int, help="feature size dimention")
    parser.add_argument("--dataset_name", default=None, type=str, help="Dataset")
    parser.add_argument("--out_channels", default=None, type=int, help="number of output channels")

    parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
    parser.add_argument("--use_persistent_dataset", action="store_true", help="use monai Dataset class")
    parser.add_argument("--persistent_cache_dir", default=None, type=str, help="persistent cache directory")
    parser.add_argument("--cache_rate", default=None, type=float, help="cache rate")

    parser.add_argument("--roi_x", default=None, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=None, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=None, type=int, help="roi size in z direction")
    parser.add_argument("--space_x", default=None, type=float, help="space size in x direction")
    parser.add_argument("--space_y", default=None, type=float, help="space size in y direction")
    parser.add_argument("--space_z", default=None, type=float, help="space size in z direction")
    parser.add_argument("--num_positive", default=1, type=int, help="numbers of positive samples for each image")
    parser.add_argument("--num_negative", default=1, type=int, help="numbers of negative samples for each image")
    parser.add_argument("--RandFlipd_prob", default=None, type=float, help='prob for the flipping')
    parser.add_argument("--RandRotate90d_prob", default=None, type=float, help='prob for the rotation')
    parser.add_argument("--RandScaleIntensityd_prob", default=None, type=float, help='prob for the intensigty scaling')
    parser.add_argument("--RandShiftIntensityd_prob", default=None, type=float, help='prob for the intensity shifting')


    parser.add_argument("--overlap", default=None, type=float, help="Overlap for inference.")
    parser.add_argument("--warmup_epochs", default=None, type=int, help="number of warmup epochs")
    parser.add_argument("--model_name", default="vit_base",
                        choices=['vit_base', 'vit_small', 'vit_tiny', 'vit_large', 'vit_huge',
                                 'swin_unetr','unet', 'segresnet',
                                 'convit3d', 'convit3d_tiny', 'MiM',
                                 'GL-MAE',
                                 'model_genesis', 'GVSL',
                                 'resnet18', 'resnet34',
                                 'swin', 'swin_tiny',
                                 'swin_unetr_tiny',
                                 'MiT',
                                 'PCRL', 'PCRLv2',
                                 'SENet154',
                                 'MoCoV2_segresnet',
                                 'jigsaw_swin', 'rubik_swin', 'positionLabel_swin',
                                 'HPM_mae_vit_base_patch16', 'localMIM_vit_base_patch16', 'Adam', 'simMIM_swin'
                                 ],
                        type=str, help="model name")

    parser.add_argument("--fold", default=None, type=int, help="Five cross validation.")
    parser.add_argument("--train_files_num", default=None, type=int, help="train number files.")

    # Print logs.
    parser.add_argument("--print_freq", default=None, type=int, help="print frequency")
    parser.add_argument("--val_every", default=None, type=int, help="validation frequency")

    parser.add_argument("--seed", default=None, type=int, help="Setting random seed.")
    parser.add_argument("--best_acc", default=None, type=int, help="Best accuracy.")

    parser.add_argument("--clip_value", default=5, type=int, help="Clip value for gradient clipping.")

    parser.add_argument('--normalize', action='store_true', help='normalize the data')

    parser.add_argument("--eval_only", action="store_true", help="eval_only")
    parser.add_argument("--pred_csv", default="result.csv", type=str, help="eval_only mode using this.")
    parser.add_argument("--pred_dir", default=None, type=str, help="eval_only mode using this.")
    parser.add_argument('--save_visualization', action="store_true", help='save visaulization results.')

    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    parser.add_argument('--ensemble_list', default=None, nargs='+', help='whether to ensemble models.')

    # model convvit3d
    parser.add_argument("--sr_ratio", default=1, type=int, help="multi scale token")

    # Linear
    parser.add_argument('--freeze_backbone', action='store_true', help='freeze backbone')

    parser.add_argument("--metric_name", default=None, type=str, help="metric name")

    # Statistics
    parser.add_argument("--excel_src_file", default=None, type=str, help="excel source file")
    parser.add_argument("--excel_dst_file", default=None, type=str, help="excel dst file")
    parser.add_argument("--excel_num_compared_methods", default=2, type=int, help="numbers of compared methods")
    parser.add_argument("--excel_num_samples", default=5, type=int, help="numbers of samples for compared methods")

    # For some datasets, such as MSD
    parser.add_argument('--ignore_label', default=None, nargs='+', type=int, help='ignore label')
    parser.add_argument('--a_min', type=float, default=None, help='min for threshold')
    parser.add_argument('--a_max', type=float, default=None, help='max for threshold')
    parser.add_argument('--b_min', type=float, default=0.0, help='min for threshold')
    parser.add_argument('--b_max', type=float, default=1.0, help='max for threshold')

    # simMIM_swin
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    return parser
