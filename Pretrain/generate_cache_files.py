"""
Generate the cache files for the datasets.
ZHUANG Jiaxin @240315
"""

import os
import sys
import warnings
import logging
import argparse
import pickle
import yaml
from monai.data import load_decathlon_datalist, PersistentDataset, DataLoader
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
)
from utils.data_utils import load_yaml, convert_dict2list
from utils.default_arguments import get_args
try:
    from utils.misc import print_with_timestamp
except Exception as e:
    print_with_timestamp = print

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
# sys.path.append('/jhcnas1/jiaxin/codes/project_02/Medical3DMAE_v2/')
# from util.data_utils_v5 import convert_dict2list, load_yaml

# from utils.data_utils import get_loader_v1


def get_loader_superpod(args, datalist):
    """
    Get the loader cache for the superpod.

    """
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
    # val_ds = PersistentDataset(data=datalist,
                            #    transform=val_transforms,
                            #    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                            #    cache_dir=args.cache_dir
                            #    )
    train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        sampler=train_sampler, drop_last=True, pin_memory=True,
        prefetch_factor=8,
    )
    # val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=None, drop_last=False, pin_memory=True)
    val_loader = None
    return train_loader, val_loader



if __name__ == '__main__':
    """
    Generate 100k cache files on Superpod.
    Command line:
        python generate_cache_files.py --dataset_split +100k --data_path /project/medimgfmod/foundation_CT/data --cache_dir /scratch/medimgfmod/cache
    """

    # parser = argparse.ArgumentParser(description='Cache the dataset')
    # parser.add_argument('--data_path', type=str, help='data directory')
    # parser.add_argument('--cache_dir', type=str, help='cache directory')
    # parser.add_argument('--dataset_split', type=str,
    #                     choices=['1k', '+2k', '+4k', '+8k', '+70k', '+100k'],
    #                     help='dataset split')
    # ####
    # parser.add_argument('--yaml_path', default='./configs/pretrain_datasets.yaml', type=str, help='yaml file path')
    # ####

    args = get_args().parse_args()
    args.yaml_path = './configs/pretrain_datasets.yaml'
    args.rank = 0
    # tf_args = parser.parse_args()
    # for k, v in vars(tf_args).items():
        # setattr(args, k, v)
    print_with_timestamp(args)

    if not args.data_path or not args.cache_dir:
        print_with_timestamp('Please specify the --data_path, and --cache_dir')
        sys.exit(-1)

    JSON_DIR = './jsons/'

    # Load the configuration file
    config = load_yaml(args.yaml_path)
    datalists = config['Pre-trainingDatasetPath']
    jsonlists = config['Pre-trainingDatasetJson']

    print(f'Loading dataset split {args.dataset_split}')
    # dataset_names = config['DatasetSplit'][args.dataset_split]
    # dataset_names = ['StonyBrookChestCT_v1', 'NLST_convert_v1']

    # dataset_names = ['NLST_convert_v1'] # p1
    dataset_names = [
        # 'BTCV',
                    #  'TCIAcovid19',
                    #  'Luna16-jx',
                    #  'stoic21',
                    #  'Flare23', #stop here
                    #  'LIDC_convert_v1',
                    #  'HNSCC_convert_v1',
                    #  'Totalsegmentator_dataset'
                     ] # /project/medimgfmod/foundation_CT/cache/100k_p3
    # dataset_names = [
        # 'CT_COLONOGRAPHY_converted_v1', #p4
    # ]

    # Complete
        # 'StonyBrookChestCT_v1'  #p5
    # dataset_names = ['OPC_convert_v1', # 100k_p2
                    #  'TCGA-HNSC_convert_v1',
                    #  'QIN_convert_v1',
                    #  'HNPC_convert_v1'
    # ] # /scratch/medimgfmod/foundation_CT/cache/100k_p2/
    datalist = []
    fail_cases = {}

    for dataset_name, jsonlist in zip(datalists, jsonlists):
        if dataset_name not in dataset_names:
            continue

        print_with_timestamp('-'*50)
        fail_cases[dataset_name] = []
        jsonpath = os.path.join(JSON_DIR, jsonlist)
        datapath = os.path.join(args.data_path, dataset_name)
        data_dict = load_decathlon_datalist(jsonpath, False, "training", base_dir=datapath)
        data_list = convert_dict2list(data_dict)
        datalist += data_list

        print_with_timestamp(f'Loading dataset dict {dataset_name} with total {len(data_dict)} files')
    print_with_timestamp(f'Total files number: {len(datalist)}')

    train_loader, val_loader = get_loader_superpod(args, datalist)

    for idx, data in enumerate(train_loader):
        if idx % 10 == 0:
            print_with_timestamp(f'Processed {idx} batches')

    with open(f'fail_cases_{args.dataset_split}.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(fail_cases, f, default_flow_style=False)
