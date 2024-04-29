"""Jiaxin ZHUANG
Modified on Apirl 29th, 2024.
"""

import torch
import monai
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
    MapTransform)

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
    elif args.dataset_name in ['10_Decathlon_Task03_Liver', '10_Decathlon_Task03_Liver_tumor',
                               '10_Decathlon_Task06_Lung',
                               '10_Decathlon_Task07_Pancreas', '10_Decathlon_Task07_Pancreas_tumor',
                               '10_Decathlon_Task08_HepaticVessel',
                               '10_Decathlon_Task09_Spleen',]:
        raise NotImplementedError #!! TODO
    elif args.dataset_name in ['10_Decathlon_Task10_Colon']:
        raise NotImplementedError #!! TODO
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
