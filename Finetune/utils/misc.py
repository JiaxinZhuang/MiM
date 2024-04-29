"""Jiaxin ZHUANG.
Modified Aug 21, 2023.
"""

import os
import random
import time
import datetime
from collections import defaultdict, deque
import warnings
import logging
import resource
from itertools import repeat
import collections.abc

import numpy as np
import torch
import torch.distributed as dist


def setup(args, rank, world_size):
    """Set up the environment.
    """
    args.distributed = True if world_size > 1 else False
    args.rank = rank
    if args.distributed:
        dist.init_process_group("nccl")
    else:
        print("Not using distributed training.")

    try:
        torch.cuda.set_device(rank)
        print_with_timestamp(f'Rank {rank} is using GPU {torch.cuda.current_device()}.')
    except Exception as e:
        print_with_timestamp(f'Rank {rank} failed to use GPU: {e}.')
    # Normalize the input.
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2**15, rlimit[1]))
    print_with_timestamp('Setting resource limit: '+str(resource.getrlimit(resource.RLIMIT_NOFILE)))
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    # resume ckpt
    if args.resume is None:
        args.resume = os.path.join(args.logdir, 'model_final.pt')
        print_with_timestamp(f'Resume ckpt set to {args.resume}')


def cleanup(args):
    '''Clean up the environment.'''
    if args.distributed:
        dist.destroy_process_group()


def calculate_time(func):
    '''Calculate the time of execution of a function.'''
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) / 60 / 60
        print(f"Function '{func.__name__}' took {execution_time:.6f} hours to execute.")
        return result
    return wrapper


def print_with_timestamp(*args, **kwargs):
    '''Print with timestamp.'''
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamped_args = [f"[{current_time}] {arg}" for arg in args]
    print(*timestamped_args, **kwargs, flush=True)

def is_dist_avail_and_initialized():
    '''Check if distributed training is available and initialized.'''
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

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
        '''Update the deque.'''
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
        '''Return the median of the deque.'''
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        '''Return the average of the deque.'''
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        '''Return the global average of the deque.'''
        return self.total / self.count

    @property
    def max(self):
        '''Return the maximum of the deque.'''
        return max(self.deque)

    @property
    def value(self):
        '''Return the last value of the deque.'''
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    '''Log the metrics.'''
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        '''Update the meters.'''
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
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        '''Synchronize the meters between processes.'''
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        '''Add a meter to the logger.'''
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        '''Log the information every print_freq.'''
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
        print_with_timestamp(f'{header} Total time: {total_time_str} ({total_time/len(iterable):.4f} s / it)')


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    '''Gather tensors from all processes, and return a list of gathered tensors.'''

    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
