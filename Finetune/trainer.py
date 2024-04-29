"""Jia-Xin ZHUANG @ 2023-8-21
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from monai.data import decollate_batch
import wandb

from utils.misc import calculate_time, print_with_timestamp, MetricLogger, SmoothedValue, distributed_all_gather


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    '''Training
    '''
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}/{args.max_epochs}]'

    for idx, batch_data in enumerate(metric_logger.log_every(loader, args.print_freq, header)):
        data, target = batch_data["image"].cuda(), batch_data["label"].cuda()
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)

        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # accumate for some iterations and update
        loss /= args.accum_iter
        if (idx+1) % args.accum_iter == 0:
            if args.amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            for param in model.parameters():
                param.grad = None

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print_with_timestamp(f"Loss is {loss_value}, stopping training, stopped at Epoch {epoch}/{idx}")
            sys.exit(-1)

        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            loss_value = np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0)
        metric_logger.update(train_loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

    for param in model.parameters():
        param.grad = None

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_with_timestamp("Averaged stats:", metric_logger)
    return metric_logger.meters['train_loss'].global_avg


def val_epoch(model, loader, acc_func,
              args, epoch, model_inferer=None, post_label=None, post_pred=None):
    '''Validation for segmentation.'''
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('acc', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'=>Val Epoch: [{epoch}]'

    accs = []
    acc_func.reset()
    with torch.no_grad():
        for idx, batch_data in enumerate(metric_logger.log_every(loader, args.print_freq, header)):
            data, target = batch_data["image"].cuda(), batch_data["label"].cuda()
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)

            if post_label:
                val_labels_list = decollate_batch(target)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            else:
                val_labels_convert = target
            if post_pred:
                val_outputs_list = decollate_batch(logits)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            else:
                val_output_convert = logits
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)

            if args.distributed:
                acc = acc.cuda()
                acc_list = distributed_all_gather([acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                accs.extend(acc_list)
            else:
                acc_list = acc.detach().cpu().numpy()
                if not isinstance(acc_list, list):
                    acc_list = [acc_list]
                accs.extend(acc_list)

            avg_acc = np.mean([np.nanmean(l) for l in accs]) * 100
            metric_logger.update(acc=avg_acc)

    acc_func.reset()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_with_timestamp(f"{args.model_name} on fold {args.fold} of the {args.dataset_name}. Averaged stats: {metric_logger}.")
    return avg_acc


@calculate_time
def run_training(model, train_loader, val_loader, optimizer, loss_func, acc_func,
                 args, model_inferer=None, scheduler=None,
                 start_epoch=0, post_label=None, post_pred=None):
    '''Run training.'''
    if args.logdir and args.rank == 0:
        wandb_writer = wandb
    else:
        wandb_writer = None
    scaler = GradScaler() if args.amp else None

    val_acc_max = 0.0 if args.best_acc is None else args.best_acc
    hist_epoch_secs = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch,
            loss_func=loss_func, args=args
        )
        # Update the history epoch time.
        if epoch == 0 or epoch == 1:
            # The first epoch is usually slower than the rest, so reset it.
            hist_epoch_secs = time.time() - epoch_time
        else:
            hist_epoch_secs = (time.time() - epoch_time) * 0.9 + hist_epoch_secs * 0.1

        if args.rank == 0:
            train_log = f'=> Remaining: {hist_epoch_secs*(args.max_epochs-epoch-1)/3600:.2f} h to finish training.'
            print_with_timestamp(train_log)
            if wandb_writer:
                lr = optimizer.param_groups[0]["lr"]
                wandb_writer.log({"train_loss": train_loss, "lr": lr})

        # Validation.
        if (epoch + 1) % args.val_every == 0 or epoch == 0:
            if args.distributed:
                dist.barrier()
            print_with_timestamp('Start validation.')
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                acc_func=acc_func,
                model_inferer=model_inferer,
                epoch=epoch,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
            if args.distributed:
                dist.barrier()
            if args.rank == 0:
                # use a barrier to make sure training is done on all ranks
                if wandb_writer:
                    wandb_writer.log({"val_acc": val_avg_acc})
                if val_avg_acc > val_acc_max:
                    print_with_timestamp(f'new best ({val_acc_max:.6f} --> {val_avg_acc:.6f}).')
                    val_acc_max = val_avg_acc
                    # Save best model checkpoint.
                    if args.logdir:
                        save_checkpoint(model, epoch, args, best_acc=val_acc_max,
                                        optimizer=optimizer, scheduler=scheduler,
                                        loss_scaler=scaler)
                # Save final model checkpoint.
                print_with_timestamp(f'Current acc: {val_avg_acc:.6f}. Best acc:{val_acc_max:.6f}.')
                save_checkpoint(model, epoch, args, best_acc=val_acc_max,
                                optimizer=optimizer, scheduler=scheduler,
                                loss_scaler=scaler, filename="model_final.pt")
        if scheduler:
            scheduler.step()

    print_with_timestamp(f"Training Finished !, Best Accuracy: {val_acc_max}")
    return val_acc_max


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0,
                    optimizer=None, scheduler=None, loss_scaler=None):
    '''Save checkpoint.'''
    state_dict = model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    if loss_scaler is not None:
        save_dict['loss_scaler'] = loss_scaler.state_dict()
    if args is not None:
        save_dict["args"] = args

    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print_with_timestamp("Saving checkpoint", filename)
