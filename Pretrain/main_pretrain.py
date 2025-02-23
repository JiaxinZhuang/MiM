"""Jia-Xin ZHUANG.
Based on Medical3DMAE_ST_v2.3
"""
import warnings
import logging
import resource
import datetime
import json
import os
import time
from thop import profile, clever_format
import torch
import torch.distributed as dist
# from torch.utils.tensorboard import SummaryWriter
import wandb

import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

# import models_convmae3d_v2
from networks.get_models import build_models
from engine_pretrain import train_one_epoch
from utils.data_utils import get_loader
import utils.misc as misc
from utils.misc import print_with_timestamp
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.default_arguments import get_args

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print_with_timestamp('Setting resource limit: '+str(resource.getrlimit(resource.RLIMIT_NOFILE)))


def main(rank, world_size, args):
    """Main function.
    """
    misc.setup(args, rank, world_size)
    if misc.is_main_process():
        print_with_timestamp(args)
    device = torch.device(args.device)

    # Log writer.
    if args.rank == 0 and args.logdir is not None:
        os.makedirs(args.logdir, exist_ok=True)
        # log_writer = SummaryWriter(log_dir=args.logdir)
    # else:
        # log_writer = None

    # define the model
    model, model_ema = build_models(args)

    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                                        #   find_unused_parameters=True)
                                                          find_unused_parameters=False)
        model_without_ddp = model.module
    else:
        model.cuda()
        print_with_timestamp('Using single gpu.')
        model_without_ddp = model

    if args.rank == 0:
        print_with_timestamp(f"Model = {model_without_ddp}")

    # Print model parameters.
    pytorch_total_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 10**6
    print_with_timestamp(f"Rank:{args.rank} Total parameters count: {pytorch_total_params} M")
    # if args.rank == 0 and args.model_name not in ['MoCoV2', 'Adam', 'simMIM_swin']:
        # print_with_timestamp(f"Rank:{args.rank} Total parameters count: {pytorch_total_params} M")
        # dummy = torch.randn(1, 1, 96, 96, 96).cuda()
        # if args.model_name in ['MoCoV2', 'Adam']:
            # dummy_a = torch.randn(2, 1, 96, 96, 96).cuda()
            # dummy_b = torch.randn(2, 1, 96, 96, 96).cuda()
            # macs, params = profile(model, inputs=(dummy_a, dummy_b))
        # if args.model_name == 'HPM_mae_vit_base_patch16':
            # mask = torch.zeros(1, 1, 6, 6, 6).flatten(1).to(torch.bool).cuda()
            # macs, params = profile(model, inputs=(dummy, mask))
        # else:
            # macs, params = profile(model, inputs=(dummy,))
        # macs, params = clever_format([macs, params], "%.3f")
        # print_with_timestamp(f"macs: {macs}, params: {params}")

    # Dataset.
    eff_batch_size = args.batch_size * args.accum_iter * world_size * args.sw_batch_size
    args.lr = args.blr * eff_batch_size / 256
    if args.rank == 0:
        print_with_timestamp(f"accumulate grad iterations: {args.accum_iter}")
        print_with_timestamp(f"effective batch size: {eff_batch_size}")
        print_with_timestamp(f"actual lr: {args.lr:.5f}")

    train_loader = get_loader(args)

    # Loss.
    loss_scaler = NativeScaler()

    # Optimizer.
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    if not args.test_one_epoch:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, model_ema=model_ema.ema if model_ema else None, optimizer=optimizer, loss_scaler=loss_scaler)
        misc.resume_ckpt(args, model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    else:
        print_with_timestamp('Test one epoch only.')

    if args.rank == 0:
        print_with_timestamp(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.test_one_epoch and epoch >= 1:
            print('Test one epoch completed.')
            break

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model, model_ema=model_ema, data_loader=train_loader, optimizer=optimizer, epoch=epoch, loss_scaler=loss_scaler, log_writer=None, args=args)

        if args.distributed:
            dist.barrier()

        # save checkpoint as default
        if epoch % 100 == 0 or (epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, model_ema=model_ema,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
        # save frequency
        if epoch % args.save_fq == 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                model_ema=model_ema,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name='final')

        if args.logdir and misc.is_main_process():
            # Save the epoch each iteration.
            # log_writer.flush()
            with open(os.path.join(args.logdir, "log.txt"), mode="a",
                      encoding="utf-8") as f:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                'epoch': epoch,}
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

    misc.cleanup(args)


if __name__ == '__main__':
    args = get_args().parse_args()

    exp_name = os.path.basename(args.logdir)
    project_name = 'project02'

    WORLD_SIZE = torch.cuda.device_count()
    if WORLD_SIZE == 1:
        print_with_timestamp('Using single gpu.')
        wandb.init(
            project=project_name,
            name=exp_name,
            config=vars(args),
            tags=args.tags,
            resume=True
        )
        main(0, WORLD_SIZE, args)
    else:
        print_with_timestamp('Using multiple gpus.')
        args.rank = int(os.environ["LOCAL_RANK"])
        print_with_timestamp(f'rank: {args.rank}')
        if args.rank == 0:
            wandb.init(
                project=project_name,
                name=exp_name,
                config=vars(args),
                tags=args.tags,
                resume=True
            )
        main(args.rank, WORLD_SIZE, args)
