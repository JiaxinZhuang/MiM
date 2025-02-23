"""
Last Modified: 2023-12-03
For [~1k, ~2k, ~4k, ~8k, ~70k, ~100k] large scale CT dataset.
Jiaxin ZHUANG.
"""

import pickle
import os
import yaml
import torch.distributed as dist

from monai.data import DataLoader, DistributedSampler, load_decathlon_datalist, PersistentDataset
from monai.transforms import (
    LoadImage,
    Orientation,
    ScaleIntensityRange,
    ToTensor,
    Compose,
    CropForeground,
    RandSpatialCropSamples,
    SpatialPad,
)

try:
    from utils.misc import print_with_timestamp
except Exception as e:
    print_with_timestamp = print


def get_loader(args):
    '''Get dataset loader.'''
    if args.dataset_loader == 'v1':
        train_loader = get_loader_v1(args)
        print_with_timestamp('Using dataset. with get_loader v1')
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
            if dataset_name == 'NLST_convert_v1':
                data_dict = load_decathlon_datalist(jsonpath, False, "training", base_dir='/data/CT/data/NLST_convert_v1')
                if args.rank == 0:
                    print_with_timestamp('=> Using NLST_convert_v1 dataset, loading from /data/CT/data')
            else:
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

    if args.rank == 0:
        print('Using MONAI Persistent Dataset.')
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
    return train_loader