"""Jiaxin ZHUANG
Modified on Apirl 29th, 2024.
"""

import os
from functools import partial
import wandb
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data.distributed
import torch.multiprocessing

from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activations, Compose

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from networks.net import get_model
from utils.downstream_utils import load_ckpt, resume_ckpt
from utils.helper import load_config_yaml_args
from utils.downstream_data_utils import get_loader
from utils.misc import print_with_timestamp, setup, cleanup
from utils.default_arguments import get_args
from utils.loss import get_loss_function
from utils.metric import get_metric


def main(rank, world_size, args):
    """Main function.
    """
    setup(args, rank, world_size)
    print_with_timestamp(args)
    device = torch.device(args.device)

    # Dataset.
    train_loader, val_loader, task = get_loader(args)
    args.task = task
    print_with_timestamp(f'Setting task to {args.task}')

    # Model.
    model = get_model(args)
    model.to(device)
    if args.distributed:
        print_with_timestamp(torch.cuda.current_device())
        model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=False)
        print_with_timestamp(f'{args.rank} Using DDP.')
        model_without_ddp = model.module
    else:
        model.cuda()
        print_with_timestamp('Using single gpu.')
        model_without_ddp = model

    #!!TODO
    # if args.task not in ['cls'] and args.model_name not in ['MiT', 'PCRLv2', 'swin', 'swin_unetr_tiny', 'MoCoV2_segresnet', 'jigsaw_swin', 'rubik_swin', 'positionLabel_swin', 'localMIM_vit_base_patch16', 'simMIM_swin', 'Adam'] and args.pretrained_path is not None:
        # model = load_ckpt(args, model_without_ddp)
    model_list = [
        'vit_base', 'GL-MAE', 'convit3d', 'swin_unetr', 'localMIM_vit_base_patch16',
        'HPM_mae_vit_base_patch16',
    ]
    if args.task in ['seg'] and args.model_name in model_list and args.pretrained_path is not None:
        model = load_ckpt(args, model_without_ddp)

    # Print model parameters.
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6
    print_with_timestamp(f"Rank:{args.rank} Total parameters count: {pytorch_total_params} M")

    # Loss.
    loss_func = get_loss_function(args)

    # Optimizer.
    eff_batch_size = args.batch_size * args.sw_batch_size * world_size * args.accum_iter
    args.optim_lr = args.optim_lr * eff_batch_size / 16
    if args.rank == 0:
        print_with_timestamp(f'Effective batch size: {eff_batch_size}, learning rate: {args.optim_lr}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr,
                                  weight_decay=args.reg_weight)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
    )

    model, optimizer, scheduler = resume_ckpt(args, model, optimizer, scheduler)

    # Metric.
    if args.dataset_name == 'CC_CCII':
        post_label = None
        post_pred = None
        model_inferer = None
    elif args.dataset_name == '10_Decathlon_Task01_BrainTumour':
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(argmax=False, threshold=0.5)])
        post_label = None
        inf_size = [args.roi_x, args.roi_y, args.roi_z]
        model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=2,
            predictor=model,
            overlap=args.overlap,
        )
    else:
        inf_size = [args.roi_x, args.roi_y, args.roi_z]
        model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=2,
            predictor=model,
            overlap=args.overlap,
        )
        post_label = AsDiscrete(to_onehot=args.out_channels)
        post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)

    # Get metric
    acc = get_metric(args)

    # Resume training.
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=loss_func,
        acc_func=acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=args.start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    cleanup(args)


if __name__ == "__main__":
    args = get_args().parse_args()
    load_config_yaml_args(args.config_path, args)
    args.amp = not args.noamp
    args.model_name = 'convvit3d' if args.model_name == 'MiM' else args.model_name

    # set the wandb project where this run will be logged
    wandb.init(
        project=args.project_name,
        name=os.path.basename(args.logdir),
        config=vars(args),
        resume=True,
    )

    if args.amp:
        print_with_timestamp('Training with amp')
    else:
        print_with_timestamp('Training without amp')
    WORLD_SIZE = torch.cuda.device_count()

    if WORLD_SIZE == 1 or WORLD_SIZE == 0:
        print_with_timestamp('Using single gpu.')
        main(0, WORLD_SIZE, args)
    else:
        print_with_timestamp('Using multiple gpus.')
        args.rank = int(os.environ["LOCAL_RANK"])
        print_with_timestamp(f'rank: {args.rank}')
        main(args.rank, WORLD_SIZE, args)

    wandb.finish()
