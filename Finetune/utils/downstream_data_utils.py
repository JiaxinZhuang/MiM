"""
Author: Jiaxin ZHUANG
Date: 2025-07-08
"""

import torch
import numpy as np
from copy import deepcopy
import monai
from monai import data
from monai.data import load_decathlon_datalist, PersistentDataset
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    RandAffined,
    EnsureTyped,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord,
    SpatialPadd,
    RandZoomd,
    RandGaussianNoised,
    CastToTyped,
    RandRotate90d,
    RandCropByLabelClassesd,
    MapTransform,
    RandZoomdSelect,
    Sampler,
    DataLoader,
    )

try:
    from utils.misc import print_with_timestamp
except ImportError:
    print_with_timestamp = print


def get_loader(args):
    '''Get the dataloader for the downstream segmentation task.'''
    task = None
    if args.dataset_name == 'input_file' and args.eval_only:
        raise NotImplementedError #!! TODO
    elif args.dataset_name == 'BTCV':
        raise NotImplementedError #!! TODO
    elif args.dataset_name == 'MMWHS_ct':
        raise NotImplementedError #!! TODO
    elif args.dataset_name == 'msd_spleen':
        raise NotImplementedError #!! TODO
    elif args.dataset_name in ['10_Decathlon_Task03_Liver',
                               '10_Decathlon_Task03_Liver_tumor',
                               '10_Decathlon_Task06_Lung',
                               '10_Decathlon_Task07_Pancreas',
                               '10_Decathlon_Task07_Pancreas_tumor',
                               '10_Decathlon_Task08_HepaticVessel',
                               '10_Decathlon_Task09_Spleen',
                               '10_Decathlon_Task10_Colon',
                               ]:
        data = get_seg_loader_10_decathlon(args)
        task = 'seg'
    elif args.dataset_name == 'Covid19_20':
        data = get_seg_loader_covid19(args)
        task = 'seg'
    elif args.dataset_name == 'CC_CCII':
        raise NotImplementedError #!! TODO
    elif args.dataset_name in ['Flare22', 'Amos22']:
        raise NotImplementedError #!! TODO
    elif args.dataset_name in ['10_Decathlon_Task01_BrainTumour', '10_Decathlon_Task02_Heart', '10_Decathlon_Task05_Prostate']:
        raise NotImplementedError #!! TODO
    else:
        raise NotImplementedError #!! TODO
    return [*data, task]


def load_fold(args, datalist_json, data_dir):
    '''Load the fold of the dataset.'''
    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    if args.fold is not None:
        train_files = []
        val_files = []
        for dd in datalist:
            if dd["fold"] != args.fold:
                train_files.append(dd)
            else:
                val_files.append(dd)
    else:
        train_files = datalist
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)

    if args.rank == 0:
        print_with_timestamp(f"train_files: {len(train_files)}")
        print_with_timestamp(f"val_files: {len(val_files)}")
        print_with_timestamp(val_files)
    return train_files, val_files


def get_seg_loader_covid19(args):
    """Get the dataloader for the Covid19-20.
    """
    def _get_xforms(mode="train", keys=("image", "label")):
        """returns a composed transform for train/val/infer."""

        xforms = [
            LoadImaged(keys, ensure_channel_first=True, image_only=True),
            Orientationd(keys, axcodes="LPS"),
            Spacingd(keys, pixdim=(1.25, 1.25, 5.0),
                     mode=("bilinear", "nearest")[: len(keys)]),
            ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0,
                                 b_min=0.0, b_max=1.0, clip=True),
        ]
        if mode == "train":
            xforms.extend(
                [
                    SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 96x96x96
                    RandAffined(
                        keys,
                        prob=0.15,
                        rotate_range=(0.05, 0.05, None),
                        scale_range=(0.1, 0.1, None),
                        mode=("bilinear", "nearest"),
                    ),
                    SpatialPadd(keys, spatial_size=(192, 192, -1 if args.roi_z == 16 else args.roi_z), mode="constant"),
                    # ensure at least 96x96x96
                    RandCropByPosNegLabeld(keys, label_key=keys[1],
                                           spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                           num_samples=args.sw_batch_size),
                    RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                    RandFlipd(keys, spatial_axis=0, prob=0.5),
                    RandFlipd(keys, spatial_axis=1, prob=0.5),
                    RandFlipd(keys, spatial_axis=2, prob=0.5),
                ]
            )
            dtype = (torch.float32, torch.uint8)
        if mode == "val":
            dtype = (torch.float32, torch.uint8)
        xforms.extend([CastToTyped(keys, dtype=dtype), EnsureTyped(keys)])
        return monai.transforms.Compose(xforms)

    train_files, val_files = load_fold(args=args, data_dir=args.data_dir,
                                       datalist_json=args.json_list)
    keys = ("image", "label")
    train_transforms = _get_xforms("train", keys)
    train_ds = PersistentDataset(data=train_files, transform=train_transforms,
                                 cache_dir=args.persistent_cache_dir)
    train_loader = monai.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    val_transforms = _get_xforms("val", keys)
    val_ds = PersistentDataset(data=val_files, transform=val_transforms,
                               cache_dir=args.persistent_cache_dir)
    val_loader = monai.data.DataLoader(
        val_ds, batch_size=1, pin_memory=True,
        num_workers=args.num_workers, shuffle=False
    )

    if args.eval_only:
        return val_loader, val_files
    else:
        return train_loader, val_loader


def get_seg_loader_10_decathlon(args):
    """
    Get the dataloader for the following datasets:
        10_Decathlon_Task03_Liver, 10_Decathlon_Task03_Liver_tumor,
        10_Decathlon_Task06_Lung,
        10_Decathlon_Task07_Pancreas, 10_Decathlon_Task07_Pancreas_tumor,
        10_Decathlon_Task08_HepaticVessel,
        10_Decathlon_Task09_Spleen,
        10_Decathlon_Task10_Colon,
    """
    train_files, val_files = load_fold(args=args, data_dir=args.data_dir,
                                       datalist_json=args.json_list)
    if args.dataset_name == '10_Decathlon_Task09_Spleen':
        print_with_timestamp('Using spleen dataset')
        train_transforms = Compose(
            [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                     mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.sw_batch_size,
                image_key="image",
                image_threshold=0,
            ),
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                rotate_range=(0, 0, np.pi/15),
                scale_range=(0.1, 0.1, 0.1)),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(args.roi_x, args.roi_y, args.roi_z),
                        mode=("bilinear", "nearest")),
            ]
        )
    else:
        # Train transforms
        train_transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            FilterLabels(args=args),
            SortLabelMap(args=args),
            Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandZoomdSelect(args),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                        mode='constant'),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=args.num_positive,
                neg=args.num_negative,
                num_samples=args.sw_batch_size,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                rotate_range=(0, 0, np.pi/15),
                scale_range=(0.1, 0.1, 0.1)),
            ToTensord(keys=["image", "label"]),
        ])
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                FilterLabels(args=args),
                SortLabelMap(args=args),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                        mode=("bilinear", "nearest")),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

    train_ds = PersistentDataset(data=train_files, transform=train_transforms,
                                cache_dir=args.persistent_cache_dir)
    val_ds = PersistentDataset(data=val_files, transform=val_transforms,
                               cache_dir=args.persistent_cache_dir)
    print_with_timestamp(f'Using persistent dataset: {args.persistent_cache_dir}')

    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
    )
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
        sampler=val_sampler, pin_memory=True, persistent_workers=True)
    if args.eval_only:
        # batch size must be 1 for inference!
        return val_loader, val_files
    return train_loader, val_loader


class FilterLabels(MapTransform):
    """Filter unsed label.
    """
    def __init__(self, args):
        '''Init'''
        # super().__init__()
        self.args = args

    def __call__(self, data):
        if hasattr(self.args, 'ignore_label'):
            if self.args.ignore_label is not None:
                label = deepcopy(data['label'])
                for key in self.args.ignore_label:
                    label = torch.where(label == key, torch.zeros_like(label), label)
                data['label'] = label
        return data

class SortLabelMap(MapTransform):
    """Sort the label map.
    """
    def __init__(self, args):
        '''Init'''
        self.args = args

    def __call__(self, data):
        unique_labels = torch.unique(data['label'])
        unique_labels = unique_labels[unique_labels != 0]
        result = torch.zeros_like(data['label'])
        for i, label in enumerate(unique_labels, start=1):
            result = torch.where(data['label'] == label, i, result)
        unique_labels_after = torch.unique(result)
        unique_labels_after = unique_labels_after[unique_labels_after != 0]
        data['label'] = result
        return data