"""Jia-Xin ZHUANG @ 2023-06-23
"""

import os
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from util.utils import distributed_all_gather
from monai.data import decollate_batch


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0 and idx % args.print_freq == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, acc_func, acc_func_each_class, args, 
              model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    accs = []
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            if post_label:
                val_labels_convert = [post_label(val_label_tensor).cpu() for val_label_tensor in val_labels_list]
            else:
                val_labels_convert = [val_label_tensor.cpu() for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor).cpu() for val_pred_tensor in val_outputs_list]

            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            # acc_func_each_class(y_pred=val_output_convert, y=val_labels_convert)

            acc = acc.cuda(args.rank)
            if args.distributed:
                acc_list = distributed_all_gather([acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                # avg_acc = np.mean([np.nanmean(l) for l in acc_list])
                accs.extend(acc_list)
            else:
                acc_list = acc.detach().cpu().numpy()
                # avg_acc = np.mean([np.nanmean(l) for l in acc_list])
                accs.extend(acc_list)

        avg_acc = np.mean([np.nanmean(l) for l in accs])
        # metric_batch = acc_func_each_class.aggregate()
        # if args.rank == 0:
            # output_string = ''
            # output_string_pattern = '{:.2f}'
            # print('Mean Val Dice: {:.5f}'.format(avg_acc))
            # print('----------- Each Class Dice ----------')
            # mean_dice = []
            # for index, value in enumerate(metric_batch[0]):
                # print('{}: {:.5f}'.format(index, value.item()))
                # if index == 0:
                    # output_string = output_string_pattern.format(value.item()*100) 
                # else:
                    # output_string = output_string + ' & ' + \
                        # output_string_pattern.format(value.item()*100)
                    # mean_dice.append(value.item() * 100)
            # print(output_string)
            # mean_dice = np.mean(mean_dice)
            # print('Mean Dice (Mean of each class): {:.2f}'.format(mean_dice))

    # acc_func_each_class.reset()
    acc_func.reset()
    return avg_acc


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, 
                    optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(model, train_loader, val_loader, optimizer, loss_func, acc_func,
                 acc_func_each_class, args, model_inferer=None, scheduler=None,
                 start_epoch=0, post_label=None, post_pred=None):
    writer = None
    if args.logdir and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        print("Writing Tensorboard logs to ", args.logdir)
    scaler = GradScaler() if args.amp else None
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        # Training.
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, 
            loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            if writer:
                writer.add_scalar("train_loss", train_loss, epoch)

        # Validation.
        if (epoch + 1) % args.val_every == 0 or (epoch == 0):
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func_each_class=acc_func_each_class,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, 
                                                                  val_avg_acc))
                    val_acc_max = val_avg_acc
                    # Save best model checkpoint.
                    if args.logdir and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, 
                            optimizer=optimizer, scheduler=scheduler
                        )
                # Save final model checkpoint.
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, 
                                filename="model_final.pt")
        if scheduler:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)
    return val_acc_max


def run_training_msd_task01_brainTS(args, model, train_loader, val_loader, 
                                    optimizer, lr_scheduler, loss_function, 
                                    scaler, post_trans, dice_metric, 
                                    dice_metric_batch, model_inferer):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)

    val_acc_max = 0.0
    total_start = time.time()
    for epoch in range(args.max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.max_epochs}")
        model.train()

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, 
            loss_func=loss_function, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_start),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)

        if epoch % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            model.eval()
            epoch_start = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func_each_class=dice_metric_batch,
                acc_func=dice_metric,
                model_inferer=model_inferer,
                args=args,
                post_label=None,
                post_pred=post_trans)
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_start),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, 
                                                                  val_avg_acc))
                    val_acc_max = val_avg_acc
                    # b_new_best = True
                    if args.rank == 0 and not args.logdir and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, 
                            optimizer=optimizer, scheduler=lr_scheduler
                        )
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, 
                                filename="model_final.pt")

        lr_scheduler.step()
    total_time = time.time() - total_start
    print("Training Finished !, Best Accuracy: {:.2f} \
        for {} hours", val_acc_max* 100, total_time/60/60)
    return val_acc_max