"""
Last Modified: 2023-10-03
For [~1k, ~2k, ~4k, ~8k, ~70k, ~100k] large scale CT dataset.
Jiaxin ZHUANG.
"""

import pickle
import os
import sys
import yaml
import torch
import torch.distributed as dist
import numpy as np

import monai
from monai.data import DataLoader, Dataset, DistributedSampler, load_decathlon_datalist, PersistentDataset, CacheNTransDataset
from monai.transforms import (
    LoadImage,
    Orientation,
    ScaleIntensityRange,
    ToTensor,
    Compose,
    CropForeground,
    RandSpatialCropSamples,
    SpatialPad,
    RandSpatialCrop,
    MapTransform
)

try:
    from utils.misc import print_with_timestamp
except Exception as e:
    print_with_timestamp = print


def get_loader(args):
    '''Get dataset loader.'''
    if args.dataset_loader == 'v2':
        train_loader = get_loader_v2(args)
        print_with_timestamp('Using dataset. with get_loader_v2')
    elif args.dataset_loader == 'v3':
        train_loader = get_loader_v3(args)
        print_with_timestamp('Using dataset. with get_loader_v3')
    elif args.dataset_loader == 'v1_slurm':
        from utils.data_utils_slurm import get_loader_v1_slurm
        train_loader, _ = get_loader_v1_slurm(args)
    elif args.dataset_loader == 'v1':
        train_loader, _ = get_loader_v1(args)
        print_with_timestamp('Using dataset. with get_loader v1')
    elif args.dataset_loader == 'mmsmae':
        train_loader = get_loader_mmsmae(args)
        print_with_timestamp('Using dataset. with get_loader_mmsmae')
    elif args.dataset_loader == 'MoCoV2':
        train_loader = get_loader_mocov2(args)
        print_with_timestamp('Using dataset. with get_loader_mocov2')
    elif args.dataset_loader == 'Adam':
        train_loader = get_loader_adam(args)
        print_with_timestamp('Using dataset. with get_loader_adam')
    elif args.dataset_loader == 'HPM':
        train_loader, _ = get_loader_hpm(args)
        print_with_timestamp('Using dataset. with get_loader_hpm')
    elif args.dataset_loader == 'simMIM':
        train_loader, _ = get_loader_simMIM(args)
        print_with_timestamp('Using dataset. with get_loader_simMIM')
    elif args.dataset_loader == 'GVSL':
        train_loader, _ = get_loader_GVSL(args)
        print_with_timestamp('Using dataset. with get_loader_GVSL')
    else:
        raise NotImplementedError(f'Not implemented dataset loader: {args.dataset_loader}')
    return train_loader

def is_dist_avail_and_initialized():
    '''Check if distributed training is available and initialized.'''
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    '''Get the number of processes participating in distributed training.'''
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    '''Get the rank of the current process in the global process group.'''
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def load_yaml(yaml_path):
    """Load the yaml file
    """
    with open(yaml_path, 'r', encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

def convert_dict2list(dict_list):
    '''Convert list of dict to list'''
    results = []
    for each_dict in dict_list:
        value = each_dict.values()
        results.extend(value)
    return results

def concat_dataset_split(args, dataset_splits):
    '''Concat dataset splits.'''
    # Convert to int.
    stop_key = int(args.dataset_split.replace('+', '').replace('k', ''))
    dataset_splitlists = []
    for key, value in dataset_splits.items():
        # Convert to int.
        key = int(key.replace('+', '').replace('k', ''))
        if key > stop_key:
            if args.rank == 0:
                print_with_timestamp(f'=> Skip dataset {key} with {len(value)} dataset')
            continue
        dataset_splitlists.extend(list(value))
    return dataset_splitlists


def get_dataset(args):
    '''Get dataset split.'''
    yaml_config = load_yaml(args.yaml_path)
    datalists = yaml_config['Pre-trainingDatasetPath']
    jsonlists = yaml_config['Pre-trainingDatasetJson']
    dataset_splits = yaml_config['DatasetSplit']
    dataset_splitlists = concat_dataset_split(args, dataset_splits)
    if args.rank == 0:
        print_with_timestamp(f'=> Pre-training dataset contains {len(dataset_splitlists)} datasets, and dataset_splitlists:{dataset_splitlists}')

    datalist = []
    for dataset_name, jsonlist in zip(datalists, jsonlists):
        if dataset_name not in dataset_splitlists:
            continue
        try:
            jsonpath = os.path.join(args.json_dir, jsonlist)
            datapath = os.path.join(args.data_path, dataset_name)
            # if dataset_name == 'NLST_convert_v1':
                # data_dict = load_decathlon_datalist(jsonpath, False, "training", base_dir='/data/CT/data/NLST_convert_v1')
                # if args.rank == 0:
                    # print_with_timestamp('=> Using NLST_convert_v1 dataset, loading from /data/CT/data')
            # else:
            data_dict = load_decathlon_datalist(jsonpath, False, "training", base_dir=datapath)

            data_list = convert_dict2list(data_dict)
            datalist += data_list
            if args.rank == 0:
                print_with_timestamp(f'Starting dataset {dataset_name} with total {len(data_list)} files')
        except Exception as expt:
            print(expt)
    if args.rank == 0:
        print_with_timestamp(f'=> Pre-training dataset all contains {len(datalist)} files')
    return datalist



def get_loader_v1(args):
    '''For pretraining dataset.'''
    datalist = get_dataset(args)
    # Transforms.
    train_transforms = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=args.a_min, a_max=args.a_max,
                b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPad(spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForeground(k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamples(
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensor(),
        ]
    )
    val_transforms = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=args.a_min, a_max=args.a_max,
                b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPad(spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForeground(k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCrop(
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                random_size=False),
            ToTensor(),
        ]
    )

    if args.rank == 0:
        print_with_timestamp(f'Using MONAI Persistent Dataset. {args.cache_dir}')
    train_ds = PersistentDataset(data=datalist,
                                 transform=train_transforms,
                                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                 cache_dir=args.cache_dir)
    val_ds = PersistentDataset(data=datalist,
                               transform=val_transforms,
                               pickle_protocol=pickle.HIGHEST_PROTOCOL,
                               cache_dir=args.cache_dir
                               )

    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = DistributedSampler(dataset=train_ds, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=train_sampler, drop_last=True, pin_memory=True,
        prefetch_factor=8,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=None, drop_last=False, pin_memory=True)
    print_with_timestamp(f'Using persistent dataset with {len(train_ds)} samples.')
    return train_loader, val_loader

def get_loader_v2(args):
    '''For pretraining dataset.'''
    # Load from yaml.
    yaml_config = load_yaml(args.yaml_path)
    datalists = yaml_config['Pre-trainingDatasetPath']
    jsonlists = yaml_config['Pre-trainingDatasetJson']
    dataset_splits = yaml_config['DatasetSplit']
    dataset_splitlists = concat_dataset_split(args, dataset_splits)

    if args.rank == 0:
        print_with_timestamp(f'=> Pre-training dataset contains {dataset_splitlists} datasets, and dataset_splitlists:{dataset_splitlists}')
    datalist_left = []
    datalist = []
    for dataset_name, jsonlist in zip(datalists, jsonlists):
        if dataset_name not in dataset_splitlists:
            continue
        try:
            jsonpath = os.path.join(args.json_dir, jsonlist)
            datapath = os.path.join(args.data_path, dataset_name)
            if dataset_name == 'NLST_convert_v1':
                data_dict = load_decathlon_datalist(jsonpath, False, "training", base_dir='/data/CT/data/NLST_convert_v1')
                if args.rank == 0:
                    print_with_timestamp('=> Using NLST_convert_v1 dataset, loading from /data/CT/data')
            else:
                data_dict = load_decathlon_datalist(jsonpath, False, "training", base_dir=datapath)

            data_list = convert_dict2list(data_dict)
            if dataset_name == 'NLST_convert_v1':
                datalist_left += data_list
            else:
                datalist += data_list

            if args.rank == 0:
                print_with_timestamp(f'Starting dataset {dataset_name} with total {len(data_list)} files')
        except Exception as expt:
            print(expt)

    if args.rank == 0:
        print_with_timestamp(f'=> Pre-training dataset all contains {len(datalist+datalist_left)} files')

    # Transforms.
    train_transforms = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True),
            SpatialPad(spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForeground(k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamples(
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensor(),
        ]
    )

    # Dataset.
    train_ds = PersistentDataset(data=datalist,
                                 transform=train_transforms,
                                 pickle_protocol=pickle.HIGHEST_PROTOCOL, cache_dir=args.cache_dir)
    train_ds_left = Dataset(data=datalist_left, transform=train_transforms)
    merge_ds = torch.utils.data.ConcatDataset([train_ds, train_ds_left])

    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = DistributedSampler(dataset=merge_ds, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(merge_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                              sampler=train_sampler, drop_last=True, pin_memory=True,
                              prefetch_factor=8)
    return train_loader

def get_loader_v3(args):
    '''For pretraining dataset.'''
    # Load from yaml.
    yaml_config = load_yaml(args.yaml_path)
    datalists = yaml_config['Pre-trainingDatasetPath']
    jsonlists = yaml_config['Pre-trainingDatasetJson']
    dataset_splits = yaml_config['DatasetSplit']
    dataset_splitlists = concat_dataset_split(args, dataset_splits)

    if args.rank == 0:
        print_with_timestamp(f'=> Pre-training dataset contains {len(dataset_splitlists)} datasets, and dataset_splitlists:{dataset_splitlists}')
    datalist = []
    datalist_left = []
    for dataset_name, jsonlist in zip(datalists, jsonlists):
        if dataset_name not in dataset_splitlists:
            continue
        try:
            jsonpath = os.path.join(args.json_dir, jsonlist)
            datapath = os.path.join(args.data_path, dataset_name)
            if dataset_name == 'NLST_convert_v1':
                data_dict = load_decathlon_datalist(args.jsonpath, False, "training", base_dir='/data/CT/data/NLST_convert_v1')
                if args.rank == 0:
                    print_with_timestamp('=> Using NLST_convert_v1 dataset, loading from /data/CT/data')
            else:
                data_dict = load_decathlon_datalist(jsonpath, False, "training", base_dir=datapath)

            data_list = convert_dict2list(data_dict)
            if dataset_name == 'NLST_convert_v1':
                datalist_left += data_list
            else:
                datalist += data_list

            if args.rank == 0:
                print_with_timestamp(f'Starting dataset {dataset_name} with total {data_list} files')
        except Exception as expt:
            print(expt)

    datalist = datalist_left + datalist
    datalist_left = []
    if args.rank == 0:
        print_with_timestamp(f'=> Pre-training dataset all contains {len(datalist_left+datalist)} files')

    # Transforms.
    train_transforms = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True, dtype=np.int16), #1
            Orientation(axcodes="RAS"), #2
            SpatialPad(spatial_size=[args.roi_x, args.roi_y, args.roi_z]), #3
            ScaleIntensityRange(
                a_min=args.a_min, a_max=args.a_max,
                b_min=args.b_min, b_max=args.b_max, clip=True
            ), #4
            CropForeground(k_divisible=[args.roi_x, args.roi_y, args.roi_z]), #5
            RandSpatialCropSamples(
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ), #6
            ToTensor(), #6
        ]
    )

    train_ds = CacheNTransDataset(data=datalist, transform=train_transforms,
                                  cache_n_trans=3, cache_dir=args.cache_dir,
                                  pickle_protocol=pickle.HIGHEST_PROTOCOL)

    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = DistributedSampler(dataset=train_ds, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=train_sampler, drop_last=True, pin_memory=True,
        prefetch_factor=8
    )
    return train_loader


def get_dataset_superpod(args):
    '''Get dataset split for Superpod.'''
    yaml_config = load_yaml(args.yaml_path)
    datalists = yaml_config['Pre-trainingDatasetPath']
    jsonlists = yaml_config['Pre-trainingDatasetJson']
    dataset_splits = yaml_config['DatasetSplit']
    dataset_names = dataset_splits[args.dataset_split]
    # dataset_splitlists = concat_dataset_split(args, dataset_splits)
    # if args.rank == 0:
        # print_with_timestamp(f'=> Pre-training dataset contains {len(dataset_splitlists)} datasets, and dataset_splitlists:{dataset_splitlists}')

    print_with_timestamp(f'=> Pre-training dataset contains {len(dataset_splits)} datasets, and dataset_splits:{dataset_splits}')

    datalist = []
    for dataset_name, jsonlist in zip(datalists, jsonlists):
        if dataset_name not in dataset_names:
            continue
        try:
            jsonpath = os.path.join(args.json_dir, jsonlist)
            datapath = os.path.join(args.data_path, dataset_name)
            data_dict = load_decathlon_datalist(jsonpath, False, "training", base_dir=datapath)

            data_list = convert_dict2list(data_dict)
            datalist += data_list
            if args.rank == 0:
                print_with_timestamp(f'Starting dataset {dataset_name} with total {len(data_list)} files')
        except Exception as expt:
            print(expt)
    if args.rank == 0:
        print_with_timestamp(f'=> Pre-training dataset all contains {len(datalist)} files')
    return datalist


def get_loader_superpod(args):
    '''For pretraining dataset used in Superpod.
    ZHUANG Jiaxin@240315
    '''
    datalist = get_dataset_superpod(args)
    # Transforms.
    train_transforms = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=args.a_min, a_max=args.a_max,
                b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPad(spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForeground(k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamples(
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensor(),
        ]
    )
    val_transforms = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=args.a_min, a_max=args.a_max,
                b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPad(spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForeground(k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCrop(
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                random_size=False),
            ToTensor(),
        ]
    )

    if args.rank == 0:
        print_with_timestamp(f'Using MONAI Persistent Dataset. {args.cache_dir}')
    train_ds = PersistentDataset(data=datalist,
                                 transform=train_transforms,
                                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                 cache_dir=args.cache_dir)
    val_ds = PersistentDataset(data=datalist,
                               transform=val_transforms,
                               pickle_protocol=pickle.HIGHEST_PROTOCOL,
                               cache_dir=args.cache_dir
                               )

    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = DistributedSampler(dataset=train_ds, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        train_sampler = None

    # train_loader = torch.utils.data.DataLoader(
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=train_sampler, drop_last=True, pin_memory=True,
        prefetch_factor=8,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=None, drop_last=False, pin_memory=True)
    print_with_timestamp(f'Using persistent dataset with {len(train_ds)} samples.')
    return train_loader, val_loader


def get_loader_mmsmae(args=None):
    '''For MMSMAE dataset.
    1. get dataset split.
    2. get dataloader with transforms.
    '''
    datalist = get_dataset(args)
    # Transforms.
    train_transforms = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True),
            #LoadImage(ensure_channel_first=True),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=args.a_min, a_max=args.a_max,
                b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPad(spatial_size=[args.crop_x, args.crop_y, args.crop_z]),
            #CropForeground(k_divisible=[args.crop_x, args.crop_y, args.crop_z]),
            RandSpatialCrop(roi_size=[args.crop_x, args.crop_y, args.crop_z], random_size=False),
            # monai.transforms.Resize(spatial_size=(192, 192, 192)), #!!TODO
            #CropForeground(k_divisible=[args.crop_x, args.up_roi_y, args.up_roi_z]),
            #RandSpatialCropSamples(
            #    roi_size=[args.roi_x, args.roi_y, args.roi_z],
            #    num_samples=args.sw_batch_size,
            #    random_center=True,
            #    random_size=False,
            #),
            ToTensor(),
        ]
    )
    train_ds = PersistentDataset(data=datalist, transform=train_transforms,
                                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                 cache_dir=args.cache_dir)
    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = DistributedSampler(dataset=train_ds, num_replicas=num_tasks,
                                           rank=global_rank, shuffle=True)
    else:
        train_sampler = None
    train_loader = monai.data.DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=train_sampler,
        drop_last=True, pin_memory=True, prefetch_factor=4)
    return train_loader


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def get_loader_mocov2(args=None):
    '''For pretraining dataset.'''
    datalist = get_dataset(args)
    # Transforms.
    train_transforms = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=args.a_min, a_max=args.a_max,
                b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPad(spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForeground(k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamples(
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensor(),
        ]
    )
    twocrop_train_transform = TwoCropsTransform(train_transforms)
    if args.rank == 0:
        print('Using MONAI Persistent Dataset.')
    train_ds = PersistentDataset(data=datalist,
                                 transform=twocrop_train_transform,
                                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                 cache_dir=args.cache_dir)
    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = DistributedSampler(dataset=train_ds, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        train_sampler = None

    # train_loader = DataLoader(
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=train_sampler, drop_last=True, pin_memory=True,
        prefetch_factor=8,
        # collate_fn=None
    )
    return train_loader

class AdamSplitVolumes(MapTransform):
    '''Split volumes for Adam.'''
    def __init__(self, args):
        self.args = args

    def __call__(self, data):
        if self.args.granularity != 0:
            patch_sx = np.random.randint(0, self.args.granularity)
            patch_sy = np.random.randint(0, self.args.granularity)
            patch_sz = np.random.randint(0, self.args.granularity)
            patch_w = int(data.shape[-3] // self.args.granularity)
            patch_h = int(data.shape[-2] // self.args.granularity)
            patch_d = int(data.shape[-1] // self.args.granularity)
            roi_start = [patch_sx * patch_w, patch_sy * patch_h, patch_sz * patch_d]
            roi_end = [(1+patch_sx) * patch_w, (1+patch_sy) * patch_h, (1+patch_sz) * patch_d]
            return data[:, roi_start[0]:roi_end[0], roi_start[1]:roi_end[1], roi_start[2]:roi_end[2]]
        else:
            return data


def get_loader_adam(args=None):
    '''For pretraining dataset for Adam.
    '''
    datalist = get_dataset(args)
    # Transforms.
    train_transforms = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=args.a_min, a_max=args.a_max,
                b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPad(spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForeground(k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamples(
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            AdamSplitVolumes(args),
            ToTensor(),
        ]
    )
    twocrop_train_transform = TwoCropsTransform(train_transforms)
    if args.rank == 0:
        print('Using MONAI Persistent Dataset.')
    train_ds = PersistentDataset(data=datalist,
                                 transform=twocrop_train_transform,
                                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                 cache_dir=args.cache_dir)

    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = DistributedSampler(dataset=train_ds, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=train_sampler, drop_last=True, pin_memory=True,
        prefetch_factor=8,
    )
    return train_loader


def get_loader_hpm(args):
    '''For pretraining dataset for HPM.
    '''
    from dataloaders.HPM import MaskTransform
    datalist = get_dataset(args)

    # Transforms.
    train_transforms = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=args.a_min, a_max=args.a_max,
                b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPad(spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForeground(k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamples(
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensor(),
            MaskTransform(args),
        ]
    )
    # val_transforms = Compose(
    #     [
    #         LoadImage(ensure_channel_first=True, image_only=True),
    #         Orientation(axcodes="RAS"),
    #         ScaleIntensityRange(
    #             a_min=args.a_min, a_max=args.a_max,
    #             b_min=args.b_min, b_max=args.b_max, clip=True
    #         ),
    #         SpatialPad(spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
    #         CropForeground(k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
    #         RandSpatialCrop(
    #             roi_size=[args.roi_x, args.roi_y, args.roi_z],
    #             random_size=False),
    #         ToTensor(),
    #     ]
    # )

    if args.rank == 0:
        print_with_timestamp(f'Using MONAI Persistent Dataset. {args.cache_dir}')
    train_ds = PersistentDataset(data=datalist,
                                 transform=train_transforms,
                                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                 cache_dir=args.cache_dir)
    # val_ds = PersistentDataset(data=datalist,
    #                            transform=val_transforms,
    #                            pickle_protocol=pickle.HIGHEST_PROTOCOL,
    #                            cache_dir=args.cache_dir
    #                            )

    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = DistributedSampler(dataset=train_ds, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=train_sampler, drop_last=True, pin_memory=True,
        prefetch_factor=8,
    )
    # val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=None, drop_last=False, pin_memory=True)
    val_loader = None
    print_with_timestamp(f'Using persistent dataset with {len(train_ds)} samples.')
    return train_loader, val_loader


def get_loader_simMIM(args):
    '''For pretraining dataset for simMIM.
    '''
    from dataloaders.simMIM import simMIM_transforms
    datalist = get_dataset(args)

    # Transforms.
    train_transforms = Compose(
        [
            LoadImage(ensure_channel_first=True, image_only=True),
            Orientation(axcodes="RAS"),
            ScaleIntensityRange(
                a_min=args.a_min, a_max=args.a_max,
                b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            SpatialPad(spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForeground(k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamples(
                roi_size=[args.roi_x, args.roi_y, args.roi_z],
                num_samples=args.sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensor(),
            simMIM_transforms(args),
        ]
    )

    if args.rank == 0:
        print_with_timestamp(f'Using MONAI Persistent Dataset. {args.cache_dir}')
    train_ds = PersistentDataset(data=datalist,
                                 transform=train_transforms,
                                 pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                 cache_dir=args.cache_dir)

    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = DistributedSampler(dataset=train_ds, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=train_sampler, drop_last=True, pin_memory=True,
        prefetch_factor=8,
    )
    val_loader = None
    print_with_timestamp(f'Using persistent dataset with {len(train_ds)} samples.')
    return train_loader, val_loader


def get_loader_GVSL(args):
    '''For pretraining dataset for simMIM.
    '''
    from dataloaders.GVSL import DatasetFromFolder3D
    datalist = get_dataset(args)
    train_ds = DatasetFromFolder3D(datalist)


    # Transforms.
    # train_transforms = Compose(
    #     [
    #         LoadImage(ensure_channel_first=True, image_only=True),
    #         Orientation(axcodes="RAS"),
    #         ScaleIntensityRange(
    #             a_min=args.a_min, a_max=args.a_max,
    #             b_min=args.b_min, b_max=args.b_max, clip=True
    #         ),
    #         SpatialPad(spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
    #         CropForeground(k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
    #         RandSpatialCropSamples(
    #             roi_size=[args.roi_x, args.roi_y, args.roi_z],
    #             num_samples=args.sw_batch_size,
    #             random_center=True,
    #             random_size=False,
    #         ),
    #         ToTensor(),
    #         simMIM_transforms(args),
    #     ]
    # )

    # if args.rank == 0:
    #     print_with_timestamp(f'Using MONAI Persistent Dataset. {args.cache_dir}')
    # train_ds = PersistentDataset(data=datalist,
    #                              transform=train_transforms,
    #                              pickle_protocol=pickle.HIGHEST_PROTOCOL,
    #                              cache_dir=args.cache_dir)

    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = DistributedSampler(dataset=train_ds, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=train_sampler, drop_last=True, pin_memory=True,
        prefetch_factor=8,
    )
    val_loader = None
    print_with_timestamp(f'Using persistent dataset with {len(train_ds)} samples.')
    return train_loader, val_loader


if __name__ == '__main__':
    import sys
    sys.path.append('/jhcnas1/jiaxin/codes/project_02/MMSMAE_20230904')
    print(sys.path)
    from main_pretrain import get_args
    from tqdm import tqdm


if __name__ == '__main__':
    import sys
    sys.path.append('/jhcnas1/jiaxin/codes/project_02/MMSMAE_20230904')
    print(sys.path)
    from main_pretrain import get_args
    from tqdm import tqdm
    # from helper import load_config_yaml_args
    args = get_args().parse_args()
    # args.config_path = '../configs/downstream_configs.yaml'

    # args.json_list = '../jsons/10_Decathlon_Task03_Liver_folds.json'
    # args.data_dir = '/data/10_Decathlon/Task03_Liver'
    # args.dataset_name = '10_Decathlon_Task03_Liver'
    # args.cache_rate = 0
    # load_config_yaml_args(args.config_path, args)
    # args.fold = 0
    # train_loader, val_loader = get_seg_loader_10_Decathlon_Task03_Liver(args)
    # for index, data in enumerate(train_loader):
    args.rank = 0
    args.distributed = False
    args.data_path= '/home/jiaxin/data'

    def test_loader_v3():
        args.dataset_split= '+100k'
        args.cache_dir= '/home/jiaxin/cache/cache_80k_20230903'
        train_loader = get_loader_v3(args)

    def test_mocov2():
        args.data_path='/home/jiaxin/data'
        args.cache_dir='/data/jiaxin/cache/1k_mocov2_231106'
        args.dataset_split='1k'
        train_loader = get_loader_mocov2(args)
        return train_loader

    train_loader = test_mocov2()
    for idx, (data) in enumerate(train_loader):
        image1 = data[0]
        image2 = data[1]
        print(image1.shape)
        print(image2.shape)

