"""Jiaxin ZHUANG
Email: lincolnz9511@gmail.com.
"""

import os
from glob import glob
import numpy as np
import torch
import math
from os.path import join
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist, PersistentDataset
from monai.transforms import (
    AddChanneld,
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
    RandRotate90d,
    ToTensord,
    SpatialPadd,
    RandZoomd,
    MapTransform)
from monai import data, transforms
from monai.data import load_decathlon_datalist
from monai.apps import DecathlonDataset
from copy import deepcopy


def get_seg_loader(args):
    if args.dataset_name == 'BTCV':
        # train_loader, val_loader = get_seg_loader_BTCV(args)
        train_loader, val_loader = get_seg_loader_BTCV_v2(args)
    elif args.dataset_name == 'msd_spleen':
        train_loader, val_loader = get_seg_loader_msd_spleen(args)
    elif args.dataset_name == 'msd_brainT':
        train_loader, val_loader = get_seg_loader_msd_brainT(args)
    elif args.dataset_name in ['10_Decathlon_Task03_Liver',
                               '10_Decathlon_Task06_Lung',
                               '10_Decathlon_Task07_Pancreas',
                               '10_Decathlon_Task08_HepaticVessel',
                               '10_Decathlon_Task09_Spleen',
                               '10_Decathlon_Task10_Colon']:
        train_loader, val_loader = get_seg_loader_10_Decathlon_Task03_Liver(args)
    else:
        raise NotImplementedError
    return train_loader, val_loader


def get_seg_loader_BTCV(args):
    """Read from json list for data split.
    """
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), 
                mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
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
        print("train_files: ", len(train_files))
        print(train_files)
        print("val_files: ", len(val_files))
        print(val_files)

    if args.use_normal_dataset:
        train_ds = data.Dataset(data=train_files, transform=train_transform)
    else:
        train_ds = data.CacheDataset(
            data=train_files, transform=train_transform, cache_num=50, cache_rate=1.0)
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
    val_ds = data.Dataset(data=val_files, transform=val_transform)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=val_sampler,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader

   
def get_seg_loader_msd_spleen(args): 
    train_images = sorted(glob(os.path.join(args.data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob(os.path.join(args.data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, \
        label_name in zip(train_images, train_labels)]
    if args.train_files_num:
        train_files, val_files = data_dicts[:args.train_files_num], data_dicts[-9:]
    else:
        train_files, val_files = data_dicts[:-9], data_dicts[-9:]
    print('train_files: ', len(train_files))
    print('val files: ', len(val_files))

    train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
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
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        # user can also add other random transforms
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
                a_min=-57,
                a_max=164,
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
    # train_ds = CacheDataset(data=train_files, transform=train_transforms, 
                            # cache_rate=0.0)
    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                            #   shuffle=True, num_workers=args.num_workers)
    # val_ds = CacheDataset(data=val_files, transform=val_transforms, 
                        #   cache_rate=0.0)
    # val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

    train_ds = data.CacheDataset(
        data=train_files, transform=train_transforms, cache_num=50, cache_rate=1.0)

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
    val_ds = data.Dataset(data=val_files, transform=val_transforms)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
        sampler=val_sampler, pin_memory=True, persistent_workers=True)
    # val_loader = data.DataLoader(
        # val_ds,
        # batch_size=1,
        # shuffle=False,
        # num_workers=args.num_workers,
        # sampler=val_sampler,
        # pin_memory=True,
        # persistent_workers=True,
        # )
    return train_loader, val_loader


def get_seg_loader_msd_brainT(args):
    train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    train_ds = DecathlonDataset(
        root_dir=args.data_dir,
        task="Task01_BrainTumour",
        transform=train_transform,
        section="training",
        download=False,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                              shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler,
                              pin_memory=True, persistent_workers=True)
    val_ds = DecathlonDataset(
        root_dir=args.data_dir,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                            sampler=val_sampler, pin_memory=True, 
                            persistent_workers=True,
                            num_workers=args.num_workers)
    return train_loader, val_loader


def get_seg_loader_BTCV_v2(args):
    """Get the dataloader for the BTCV dataset, following universal model.
    """
    data_dir = args.data_dir
    datalist_json = join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            LoadImaged(keys=["image", "label"]),
            # AddChanneld(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            ToTensord(keys=["image", "label"]),
        ]
    )
    #datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    #if args.use_normal_dataset:
    #    train_ds = data.Dataset(data=datalist, transform=train_transform)
    #else:
    #    train_ds = data.CacheDataset(
    #        data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0)
    #train_sampler = Sampler(train_ds) if args.distributed else None
    #train_loader = data.DataLoader(
    #    train_ds,
    #    batch_size=args.batch_size,
    #    shuffle=(train_sampler is None),
    #    num_workers=args.num_workers,
    #    sampler=train_sampler,
    #    pin_memory=True,
    #    persistent_workers=True,
    #)
    train_loader = None

    # val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)

    val_img = []
    val_lbl = []
    val_name = []
    with open(join(args.data_txt_path, args.dataset_list+'_val.txt'), 'r') as f:
        for line in f:
            # name = line.strip().split()[0].split('/')[0]
            name = line.strip().split()[1].split('.')[0]
            data_path, label_path = line.strip().split()
            data_path = join(args.data_dir, data_path)
            label_path = join(args.data_dir, label_path)
            val_img.append(data_path)
            val_lbl.append(label_path)
            val_name.append(name)
    val_files = [{'image': image, 'label': label, 'name': name}
                 for image, label, name in zip(val_img, val_lbl, val_name)]
    print(val_files)
    print('=>Val len {}'.format(len(val_files)))

    if args.cache_dataset:
        val_ds = data.CacheDataset(
           data=val_files, transform=val_transform, cache_num=6, cache_rate=1.0)
    else:
        val_ds = data.Dataset(data=val_files, transform=val_transform)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=val_sampler,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader, val_loader


class RandZoomd_select(RandZoomd):
    def __init__(self, args):
        self.args = args
        super().__init__(keys=["image", "label"], prob=0.3, min_zoom=1.3, 
                         max_zoom=1.5, mode=['area', 'nearest'])

    def __call__(self, data, lazy=None):
        key = self.args.dataset_name
        if key in ['10_Decathlon_Task06_Lung', '10_Decathlon_Task07_Pancreas',
                   '10_Decathlon_Task10_Colon']:
            data = super().__call__(data, lazy=lazy)
            return data
        else:
            return data

def get_seg_loader_10_Decathlon_Task03_Liver(args):
    train_files, val_files = load_fold(args=args, data_dir=args.data_dir, 
                                       datalist_json=args.json_list) 
    # Train transforms
    train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        FilterLabels(args=args),
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
        SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), 
                    mode='constant'),
        RandZoomd_select(args), 
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=2,
            neg=1,
            num_samples=args.sw_batch_size,
            image_key="image",
            image_threshold=0,
        ),
        # user can also add other random transforms
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=1.0, spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            rotate_range=(0, 0, np.pi/15),
            scale_range=(0.1, 0.1, 0.1)),
        ToTensord(keys=["image", "label"]),
    ]
    )
    # Val transforms.
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            FilterLabels(args=args),
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

    if args.use_persistent_dataset:
        train_ds = PersistentDataset(data=train_files, transform=train_transforms, 
                                    cache_dir=args.persistent_cache_dir)
        val_ds = PersistentDataset(data=val_files, transform=val_transforms, 
                                   cache_dir=args.persistent_cache_dir)
    else:
        train_ds = CacheDataset(
            data=train_files, transform=train_transforms, cache_rate=args.cache_rate)
        val_ds = data.Dataset(data=val_files, transform=val_transforms)

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
    return train_loader, val_loader


def load_fold(args, datalist_json, data_dir):
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
        print("train_files: ", len(train_files), flush=True)
        # print(train_files)
        print("val_files: ", len(val_files), flush=True)
        print(val_files, flush=True)
    return train_files, val_files


class FilterLabels(MapTransform):
    """Filter unsed label.
    """
    def __init__(self, args):
        self.args = args

    def __call__(self, data):
        # print(data['image'].shape)
        if hasattr(self.args, 'ignore_label'):
            label = deepcopy(data['label'])
            for key in self.args.ignore_label:
                label = torch.where(label == key, torch.zeros_like(label), label)
            data['label'] = label
        # print('After', data['image'].shape)
        return data

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


def convert_dict2list(dict_list):
    """Convert list of dict to list
    """
    results = []
    for dd in dict_list:
        value = dd.values()
        results.extend(value)
    return results


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch




if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from main_finetune_segmentation import get_args
    from helper import load_config_yaml_args
    args = get_args().parse_args()
    args.config_path = '../configs/downstream_configs.yaml'
    #Test BTCV
    # args.json_list = 'BTCV_folds.json'
    # args.data_dir = '/dev/shm/data/BTCV'
    # args.fold = 0
    # print(args)
    # train_loader, val_loader = get_seg_loader_BTCV(args)

    #Test 10_Decathlon_Task03
    args.json_list = '../jsons/10_Decathlon_Task03_Liver_folds.json'
    args.data_dir = '/data/10_Decathlon/Task03_Liver'
    args.dataset_name = '10_Decathlon_Task03_Liver'
    args.cache_rate = 0
    load_config_yaml_args(args.config_path, args)
    args.fold = 0
    train_loader, val_loader = get_seg_loader_10_Decathlon_Task03_Liver(args)
    # for index, data in enumerate(train_loader):
    for index, data in enumerate(val_loader):
        print(index, data['image'].shape, data['label'].shape, torch.unique(data['label']))