"""Jiaxin ZHUANG.
Modified Aug 21, 2023.
"""

import os
import random
import glob
import time
import datetime
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import torch
import torch.distributed as dist
from torch import inf


def setup(args, rank, world_size):
    """Set up the environment.
    """
    args.distributed = True if world_size > 1 else False
    args.rank = rank
    if args.distributed:
        dist.init_process_group("nccl")
    else:
        print_with_timestamp("Not using distributed training.")

    torch.cuda.set_device(rank)
    print_with_timestamp(f'Rank {rank} is using GPU {torch.cuda.current_device()}.')
    # Normalize the input.
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)

    # Seed
    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # resume ckpt
    if not args.resume:
        last_ckpt_path = os.path.join(args.logdir, 'model_final.pt')
        if os.path.exists(last_ckpt_path):
            args.resume = last_ckpt_path
        else:
            ckpt_list = glob.glob(os.path.join(args.logdir, 'checkpoint-*.pth'))
            if ckpt_list:
                args.resume = sorted(ckpt_list)[-1]
        print_with_timestamp(f'Resume ckpt set to {args.resume}')


def cleanup(args):
    if args.distributed:
        dist.destroy_process_group()


def calculate_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) / 60 / 60
        print_with_timestamp(f"Function '{func.__name__}' took {execution_time:.6f} hours to execute.")
        return result
    return wrapper


def print_with_timestamp(*args, **kwargs):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamped_args = [f"[{current_time}] {arg}" for arg in args]
    print(*timestamped_args, **kwargs, flush=True)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, name=None):
    output_dir = Path(args.logdir)
    if name is not None:
        epoch_name = str(name)
    else:
        epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / f'checkpoint-{epoch_name}.pth']
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            if model_ema is not None:
                to_save['model_ema'] = model_ema.ema.state_dict()

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.logdir,
                              tag="checkpoint-%s" % epoch_name,
                              client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])
        print_with_timestamp("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

        if 'ema_state_dict' in checkpoint:
            model_ema.load_state_dict(checkpoint['ema_state_dict'])
        print_with_timestamp("With optim & sched!")


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print_with_timestamp(log_msg.format(
                                         i, len(iterable), eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time), data=str(data_time),
                                         memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print_with_timestamp(log_msg.format(
                                         i, len(iterable), eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_with_timestamp('{} Total time: {} ({:.4f} s / it)'.format(
                             header, total_time_str, total_time / len(iterable)))


def resume_ckpt(args, model, optimizer=None, scheduler=None, loss_scaler=None):
    """Resume checkpoint from previous training.
    """
    if args.resume is not None and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        out = model.load_state_dict(ckpt["model"], strict=True)
        print_with_timestamp('Load model ckpt {}'.format(out))

        if 'optimizer' in ckpt and optimizer:
            out = optimizer.load_state_dict(ckpt["optimizer"])
            print_with_timestamp('Load optimizer: {}'.format(out))
        else:
            print_with_timestamp('No optimizer in ckpt.')

        if 'scheduler' in ckpt and scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
            print_with_timestamp('Load schduler: {}'.format(out))
        else:
            print_with_timestamp('No scheduler in ckpt.')

        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]
            print_with_timestamp('Load start epoch: {}'.format(start_epoch))
        else:
            print_with_timestamp('No epoch in ckpt.')

        if "scaler" in ckpt:
            loss_scaler.load_state_dict(ckpt["scaler"])
            print_with_timestamp("Load scaler: {}".format(loss_scaler))
        else:
            print_with_timestamp('No scaler in ckpt.')

        print_with_timestamp("=> loaded checkpoint '{}' (epoch {})".format(args.resume,
                                                                           start_epoch))
        args.start_epoch = start_epoch
        return model, optimizer, scheduler, loss_scaler
    else:
        args.start_epoch = 0
        print_with_timestamp("=> no checkpoint found at '{}'".format(args.resume))
        return model, optimizer, scheduler, loss_scaler


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x