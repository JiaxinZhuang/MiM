import monai
import glob
import os
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
from monai.data import (
    CacheDataset,
    Dataset,
    DataLoader,
    LMDBDataset,
    PersistentDataset,
    decollate_batch,
)

LMDB_cache = os.path.join('/tmp', "lmdb_cache")
Per_cache = os.path.join('/tmp', "persis")
data_path = '/data/10_Decathlon/Task09_Spleen'
train_files = sorted(glob.glob(os.path.join(data_path, "imagesTr", "*.nii.gz")))
# train_files = sorted(glob.glob(os.path.join(data_path, "labelsTr", "*.nii.gz")))
# Transforms.
roi_x=roi_y=roi_z=96
train_transforms = Compose(
    [
        LoadImage(ensure_channel_first=True, image_only=True),
        Orientation(axcodes="RAS"),
        ScaleIntensityRange(
            a_min=-1000, a_max=1000, 
            b_min=0, b_max=1, clip=True
        ),
        SpatialPad(spatial_size=[roi_x, roi_y, roi_z]),
        CropForeground(k_divisible=[roi_x, roi_y, roi_z]),
        RandSpatialCropSamples(
            roi_size=[roi_x, roi_y, roi_z],
            num_samples=4,
            random_center=True,
            random_size=False,
        ),
        ToTensor(),
    ]
)
# train_lmdb_ds = LMDBDataset(
    # data=train_files, transform=train_transforms, cache_dir=LMDB_cache, lmdb_kwargs={"map_async": True}
# )
train_persitence_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=Per_cache)
print('Done.')
