"""Jiaxin ZHUANG. @Aug 25, 2023
"""

import math
import sys
from copy import deepcopy
from typing import Iterable
import torch
import torch.nn as nn
import numpy as np
import wandb

from utils import lr_sched
from utils import misc
from utils.misc import print_with_timestamp
from utils.metric import accuracy
from utils.lr_sched import get_sin_weight_higher, get_cos_weight_lower


def train_one_epoch(args=None, model_ema=None, **kwargs):
    '''Get the train_one_epoch function.'''
    if args.model_name in ['MoCoV2']:
        return train_one_epoch_mocov2(args=args, **kwargs)
    elif args.model_name in ['Adam']:
        return train_one_epoch_adam(args=args, **kwargs)
    elif args.model_name in ['mae_vit_base_patch16', 'localMIM_vit_base_patch16', 'simMIM_swin']:
        return train_one_epoch_MAE(args=args, **kwargs)
    elif args.model_name in ['HPM_mae_vit_base_patch16']:
        return train_one_epoch_HPM(args=args, model_ema=model_ema, **kwargs)
    elif args.model_name in ['SwinUNETR']:
        return train_one_epoch_SwinUNETR(args=args, model_ema=model_ema, **kwargs)
    elif args.model_name in ['GVSL']:
        return train_one_epoch_GVSL(args=args, model_ema=model_ema, **kwargs)
    elif args.model_name in ['MiM', 'convvit3d', 'convvit3d_tiny', 'convmae_convvit_base_patch16']:
        return train_one_epoch_MiM(args=args, **kwargs)
    else:
        raise NotImplementedError

def train_one_epoch_MiM(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler, log_writer=None, args=None):
    '''Train one epoch.'''
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('coarse_weight', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('fine_weight', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 10
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    # if log_writer is not None:
        # print_with_timestamp(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples_copy = deepcopy(samples)
        samples_copy = samples_copy.cuda(non_blocking=True)
        with torch.cuda.amp.autocast():
            loss, *_ = model(samples_copy)

        if isinstance(loss, list):
            loss_value_up = loss[0].item()
            loss_value_mid = loss[1].item()
            loss_value_down = loss[2].item()
            loss_value_upmid = loss[3].item()
            loss_value_midown = loss[4].item()
            loss_term_weight = [1, 1, 1, 1e-2, 1e-2]
            # loss_term_weight = [1, 1, 1, 0, 0]
            loss_term_weight = [args.reconstruct_weight_up, args.reconstruct_weight_usual , args.reconstruct_weight_down, args.atten_weight_uu, args.atten_weight_ud]
            if args.mode == 'coarse_to_fine':
                loss_term_weight[0] = loss_term_weight[0] * get_cos_weight_lower(args=args, epoch=epoch)
                loss_term_weight[2] = loss_term_weight[2] * get_sin_weight_higher(args=args, epoch=epoch)
            elif args.mode == 'fine_to_coarse':
                loss_term_weight[0] = loss_term_weight[0] * get_sin_weight_higher(args=args, epoch=epoch)
                loss_term_weight[2] = loss_term_weight[2] * get_cos_weight_lower(args=args, epoch=epoch)

            loss = [loss_term_weight[i] * loss[i] for i in range(len(loss))]
            loss = sum(loss)
            loss_value = loss.item()
        else:
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            for param in model.parameters():
                param.grad = None

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if locals().get('loss_value_up', None) is not None:
            metric_logger.update(loss_up=loss_value_up)
            metric_logger.update(loss_mid=loss_value_mid)
            metric_logger.update(loss_down=loss_value_down)
            metric_logger.update(loss_upmid=loss_value_upmid)
            metric_logger.update(loss_midown=loss_value_midown)
            metric_logger.update(coarse_weight=loss_term_weight[0])
            metric_logger.update(fine_weight=loss_term_weight[2])

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        # Important.
        del samples, loss, samples_copy

    if log_writer is not None and args.rank == 0:
        loss_value_reduce = metric_logger.meters['loss'].global_avg
        wandb.log({'train_loss': loss_value_reduce, 'lr': lr})
        # log_writer.add_scalar('loss', loss_value_reduce, epoch)
        # log_writer.add_scalar('lr', lr, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_with_timestamp("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_mocov2(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, loss_scaler, log_writer=None, args=None):
    '''Train one epoch of
        MoCoV2
    '''
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('losses', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('top1', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('top5', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 50
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    criterion = nn.CrossEntropyLoss().cuda()

    if log_writer is not None:
        print_with_timestamp(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (images) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        imagesA, imagesB = images[0][0], images[1][0]
        imagesA, imagesB = deepcopy(imagesA), deepcopy(imagesB)
        imagesA = imagesA.cuda(non_blocking=True)
        imagesB = imagesB.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
             # compute output
            output, target = model(im_q=imagesA, im_k=imagesB)
            loss = criterion(output, target)

        loss_value = loss.item()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        metric_logger.update(top1=acc1)
        metric_logger.update(top5=acc5)
        metric_logger.update(losses=loss_value)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            for param in model.parameters():
                param.grad = None

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        # Important.
        del images, imagesA, imagesB, loss

    if log_writer is not None:
        loss_value_reduce = metric_logger.meters['loss'].global_avg
        log_writer.add_scalar('loss', loss_value_reduce, epoch)
        log_writer.add_scalar('lr', lr, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_with_timestamp(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_adam(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int, loss_scaler, log_writer=None, args=None):
    '''Train one epoch of
        Adam
    '''
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('losses', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 50
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if args.granularity == 0:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        from losses.adam import PurposivePrunerLoss
        criterion = PurposivePrunerLoss(args.sim_threshold).cuda()

    if log_writer is not None:
        print_with_timestamp(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (images) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        print()
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        imagesA, imagesB = images[0][0], images[1][0]
        imagesA, imagesB = deepcopy(imagesA), deepcopy(imagesB)
        imagesA = imagesA.cuda(non_blocking=True)
        imagesB = imagesB.cuda(non_blocking=True)
        print(imagesA.shape, imagesB.shape)

        with torch.cuda.amp.autocast():
             # compute output
            output, target, feature_similarities = model(im_q=imagesA, im_k=imagesB)
            if args.granularity == 0:
                loss = criterion(output, target)
            else:
                loss = criterion(output, target, feature_similarities, args)

        loss_value = loss.item()
        metric_logger.update(losses=loss_value)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            for param in model.parameters():
                param.grad = None

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        # Important.
        del images, imagesA, imagesB, loss

    if log_writer is not None:
        loss_value_reduce = metric_logger.meters['loss'].global_avg
        log_writer.add_scalar('loss', loss_value_reduce, epoch)
        log_writer.add_scalar('lr', lr, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_with_timestamp(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_MAE(model: torch.nn.Module, data_loader: Iterable,
                        optimizer: torch.optim.Optimizer, epoch: int, loss_scaler,
                        log_writer=None, args=None):
    '''Train one epoch of MAE'''
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    # if log_writer is not None:
        # print_with_timestamp('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples_copy = deepcopy(samples)
        samples_copy = samples_copy.cuda(non_blocking=True)
        # mask_copy = deepcopy(mask)
        # mask_copy = mask_copy.cuda(non_blocking=True)

        with torch.cuda.amp.autocast():
            # loss, *_ = model(samples_copy, mask_copy, samples_copy)
            loss, *_ = model(samples_copy)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print_with_timestamp("Loss is {}, stopping training".format(loss_value))
            sys.exit(-1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            for param in model.parameters():
                param.grad = None

        torch.cuda.synchronize()

        metric_logger.update(train_loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        # Important.
        del samples, loss, samples_copy
        # , mask, mask_copy

    # if log_writer is not None:
        # """TODO, I remove 1000x for simplicity.
        # We use epoch_1000x as the x-axis in tensorboard.
        # This calibrates different curves when batch size changes.
        # """
        loss_value_reduce = metric_logger.meters['train_loss'].global_avg
        # log_writer.add_scalar('train_loss', loss_value_reduce, epoch)
        # log_writer.add_scalar('lr', lr, epoch)
        wandb.log({'train_loss': loss_value_reduce, 'lr': lr})


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_with_timestamp("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_HPM(model: torch.nn.Module, model_ema: torch.nn.Module,
                        data_loader: Iterable,
                        optimizer: torch.optim.Optimizer, epoch: int, loss_scaler,
                        log_writer=None, args=None):
    '''Train one epoch of MAE'''
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if args.learning_loss:
        assert model_ema is not None
        if epoch < 100:
            model_ema.decay = 0.999 + epoch / 100 * (0.9999 - 0.999)
        else:
            model_ema.decay = 0.9999

    if log_writer is not None:
        print_with_timestamp('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = len(data_loader) * epoch + data_iter_step
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples_copy, bool_masked_pos = samples
        samples_copy = deepcopy(samples_copy).as_tensor()
        samples_copy = samples_copy.cuda(non_blocking=True)
        bool_masked_pos = bool_masked_pos.cuda(non_blocking=True).flatten(1).to(torch.bool)   # (N, L)
        visible_mask = torch.zeros_like(bool_masked_pos).cuda(non_blocking=True).to(torch.bool)

        with torch.cuda.amp.autocast():
            if model_ema is not None:
                with torch.no_grad():
                    outs_ema = model_ema.ema(samples_copy, mask=visible_mask)

            if args.learning_loss:
                # generate mask by predicted loss
                mask = model_ema.ema.generate_mask(outs_ema['loss_pred'], mask_ratio=args.mask_ratio,
                                                   guide=True, epoch=epoch, total_epoch=args.epochs)
                bool_masked_pos = mask.cuda(non_blocking=True).flatten(1).to(torch.bool)

            outs = model(samples_copy, mask=bool_masked_pos)

            loss_outs = model.module.forward_loss(
                    samples_copy,
                    outs['pix_pred'][:, -outs['mask_num']:],
                    outs['mask'],
                )

            if isinstance(loss_outs, dict):
                loss = loss_outs['mean']
            else:
                loss = loss_outs

            if args.learning_loss:
                loss_target = loss_outs['matrix']

                loss_learn = model.module.forward_learning_loss(
                    outs['loss_pred'][:,  -outs['mask_num']:],
                    bool_masked_pos,
                    loss_target.detach(),
                    relative=args.relative,
                )
                loss_learn_value = loss_learn.item()
                if not math.isfinite(loss_learn_value):
                    print("Loss learning is {}, skip".format(loss_learn_value))
                    sys.exit(1)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print_with_timestamp("Loss is {}, stopping training".format(loss_value))
            sys.exit(-1)

        loss = loss / accum_iter
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                                update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            for param in model.parameters():
                param.grad = None

        torch.cuda.synchronize()

        metric_logger.update(train_loss=loss_value)
        if args.learning_loss:
            metric_logger.update(loss_learn=loss_learn_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(grad_norm=grad_norm)
        # Important.
        del samples, loss, samples_copy

        if args.learning_loss:
            loss_learn_value_reduce = misc.all_reduce_mean(loss_learn_value)
        if log_writer is not None:
            """TODO, I remove 1000x for simplicity.
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            loss_value_reduce = metric_logger.meters['train_loss'].global_avg
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch)
            log_writer.add_scalar('lr', lr, epoch)
            if args.learning_loss:
                log_writer.add_scalar('train_loss_learn', loss_learn_value_reduce, it)

        if (data_iter_step + 1) >= len(data_loader):
            break


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_with_timestamp("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_SwinUNETR(model: torch.nn.Module, model_ema: torch.nn.Module,
                              data_loader: Iterable,
                              optimizer: torch.optim.Optimizer, epoch: int, loss_scaler,
                              log_writer=None, args=None):
    '''Train one epoch of MAE'''
    from dataloaders.SwinUNETR import rot_rand, aug_rand
    from losses.SwinUNETR import Loss

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    criterion = Loss(args.batch_size * args.sw_batch_size, args)

    if log_writer is not None:
        print_with_timestamp('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = len(data_loader) * epoch + data_iter_step
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples_copy = deepcopy(samples)
        x = samples.cuda(non_blocking=True)
        x1, rot1 = rot_rand(args, x)
        x2, rot2 = rot_rand(args, x)
        x1_augment = aug_rand(args, x1)
        x2_augment = aug_rand(args, x2)
        x1_augment = x1_augment
        x2_augment = x2_augment

        with torch.cuda.amp.autocast():
            rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
            rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
            rot_p = torch.cat([rot1_p, rot2_p], dim=0)
            rots = torch.cat([rot1, rot2], dim=0)
            imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
            imgs = torch.cat([x1, x2], dim=0)
            loss, losses_tasks = criterion(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)

        loss_value = loss.item()
        loss_recon = losses_tasks[2].item()

        if not math.isfinite(loss_value) or not math.isfinite(loss_recon):
            print_with_timestamp(f"Loss is {loss_value}, stopping training")
            print_with_timestamp(f"Loss recon is {loss_recon}, stopping training")
            sys.exit(-1)

        loss = loss / accum_iter
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                                update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            for param in model.parameters():
                param.grad = None

        torch.cuda.synchronize()

        metric_logger.update(train_loss=loss_value)
        metric_logger.update(train_recon_loss=loss_recon)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        # Important.
        del samples, loss, samples_copy

        if log_writer is not None:
            """TODO, I remove 1000x for simplicity.
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            loss_value_reduce = metric_logger.meters['train_loss'].global_avg
            loss_recon_value_reduce = metric_logger.meters['train_recon_loss'].global_avg
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch)
            log_writer.add_scalar('train_recon_loss', loss_recon_value_reduce, epoch)
            log_writer.add_scalar('lr', lr, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_with_timestamp("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_GVSL(model: torch.nn.Module, model_ema: torch.nn.Module,
                         data_loader: Iterable,
                         optimizer: torch.optim.Optimizer, epoch: int, loss_scaler,
                         log_writer=None, args=None):
    '''Train one epoch of GVSL'''
    from dataloaders.GVSL import AppearanceTransform, SpatialTransform
    from losses.GVSL import gradient_loss, ncc_loss, MSE

    style_aug = AppearanceTransform(local_rate=0.8,
                                    nonlinear_rate=0.9,
                                    paint_rate=0.9,
                                    inpaint_rate=0.2)

    # Data augmentation
    spatial_aug = SpatialTransform(do_rotation=True,
                                    angle_x=(-np.pi / 9, np.pi / 9),
                                    angle_y=(-np.pi / 9, np.pi / 9),
                                    angle_z=(-np.pi / 9, np.pi / 9),
                                    do_scale=True,
                                    scale_x=(0.75, 1.25),
                                    scale_y=(0.75, 1.25),
                                    scale_z=(0.75, 1.25),
                                    do_translate=True,
                                    trans_x=(-0.1, 0.1),
                                    trans_y=(-0.1, 0.1),
                                    trans_z=(-0.1, 0.1),
                                    do_shear=True,
                                    shear_xy=(-np.pi / 18, np.pi / 18),
                                    shear_xz=(-np.pi / 18, np.pi / 18),
                                    shear_yx=(-np.pi / 18, np.pi / 18),
                                    shear_yz=(-np.pi / 18, np.pi / 18),
                                    shear_zx=(-np.pi / 18, np.pi / 18),
                                    shear_zy=(-np.pi / 18, np.pi / 18),
                                    do_elastic_deform=True,
                                    alpha=(0., 512.),
                                    sigma=(10., 13.))

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print_with_timestamp('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (unlabed_img1, unlabed_img2) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = len(data_loader) * epoch + data_iter_step
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        unlabed_img1 = deepcopy(unlabed_img1)
        unlabed_img2 = deepcopy(unlabed_img2)
        # damage the image 1 for restoration
        unlabed_img1_aug = unlabed_img1.data.numpy()[0].copy()
        unlabed_img1_aug = style_aug.rand_aug(unlabed_img1_aug)
        unlabed_img1_aug = torch.from_numpy(unlabed_img1_aug[np.newaxis, :, :, :, :])

        unlabed_img1 = unlabed_img1.cuda()
        unlabed_img2 = unlabed_img2.cuda()
        unlabed_img1_aug = unlabed_img1_aug.cuda()

        # Augment the image 1, damaged image 1 and image 2
        mat, code_spa = spatial_aug.rand_coords(unlabed_img1.shape[2:])
        unlabed_img1_aug = spatial_aug.augment_spatial(unlabed_img1_aug, mat, code_spa)
        unlabed_img1 = spatial_aug.augment_spatial(unlabed_img1, mat, code_spa)
        unlabed_img2 = spatial_aug.augment_spatial(unlabed_img2, mat, code_spa)


        with torch.cuda.amp.autocast():
            res_A, warp_BA, aff_mat_BA, flow_BA = model(unlabed_img1_aug, unlabed_img2)
            loss_ncc = ncc_loss(warp_BA, unlabed_img1)
        # self.L_ncc_log.update(loss_ncc.data, unlabed_img1.size(0))
            loss_mse = MSE(res_A, unlabed_img1)
        # self.L_MSE_log.update(loss_mse.data, unlabed_img1.size(0))
            loss_smooth = gradient_loss(flow_BA)
        # self.L_smooth_log.update(loss_smooth.data, unlabed_img1.size(0))
            loss = loss_ncc + loss_mse + loss_smooth

        loss_value = loss.item()
        loss_ncc_value = loss_ncc.item()
        loss_mse_value = loss_mse.item()
        loss_smooth_value = loss_smooth.item()

        if not math.isfinite(loss_value):
            print_with_timestamp(f"Loss is {loss_value}, stopping training")
            sys.exit(-1)

        loss = loss / accum_iter
        grad_norm = loss_scaler(loss, optimizer, parameters=model.parameters(),
                                update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            for param in model.parameters():
                param.grad = None
                model.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(train_loss=loss_value)
        metric_logger.update(train_ncc_loss=loss_ncc_value)
        metric_logger.update(train_mse_loss=loss_mse_value)
        metric_logger.update(train_smooth_loss=loss_smooth_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        # Important.
        del loss, unlabed_img1, unlabed_img2, unlabed_img1_aug

        if log_writer is not None:
            """TODO, I remove 1000x for simplicity.
            We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            loss_value_reduce = metric_logger.meters['train_loss'].global_avg
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch)
            log_writer.add_scalar('lr', lr, epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_with_timestamp("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}