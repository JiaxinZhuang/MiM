# MiM
***

# 0. Requirements
* Python 3.8
* PyTorch 2.0.1
* MONAI 1.0.0
* CUDA 11.8
* cuDNN 8.5
* NVIDIA GPU with compute capability 8.6

```
pip install -r requirements.txt
```

## 1. Datasets
The details of the pretraining dataset and finetuning dataset are shown in the following figures. Since these datasets are all publicly available, you can download them from their official websites.

![Pretraining dataset](./assets/pretrained_dataset.png)

![Finetuning dataset](./assets/finetune_dataset.png)


The project contains two directories, _i.e.,_
1) Pretrain
2) Finetune

Since pre-training can be time-consuming, optimizing data loading is essential. A recommended approach is to first generate cache files, then use the MONAI DataLoader to load them. While these cache files may be large, they significantly accelerate the loading process. You can find the conversion script in

```
generate_cache_files.py
```


# 2. Pretrain
Please refer to the [Pretrain](./Pretrain) directory for the pretraining code. Make sure you have installed the requirements and downloaded the datasets.

```
bash scripts/xxx.sh
```


# 3. Finetune
Please refer to the [Finetune](./Finetune) directory for the finetuning code. Make sure you have installed the requirements and downloaded the datasets.

We provide the [pretrained model on 10k](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jzhuangad_connect_ust_hk/ElCam2XpVflPvynd9Ymss44Bl1zeKf9gOt-YqsOhMKyY2g?e=fMdhl5
) via the Onedrive link and you can finetune the model. Let's take the BTCV as an example.

```
bash scripts/BTCV/run.sh
```



## Reference
```
@article{zhuang2024mim,
  title={MiM: Mask in Mask Self-Supervised Pre-Training for 3D Medical Image Analysis},
  author={Zhuang, Jiaxin and Wu, Linshan and Wang, Qiong and Vardhanabhuti, Varut and Luo, Lin and Chen, Hao},
  journal={arXiv preprint arXiv:2404.15580},
  year={2024}
}
```

## Acknowledgement
This codebase heavily references the following projects and we thanks their efforts:

- [Masked Autoencoders](https://github.com/facebookresearch/mae): pretraining code
- [SwinUNETR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/Pretrain): data preparation and pretraining code
- [MONAI](https://github.com/Project-MONAI/MONAI): a powerful medical image analysis library

Wellcomes to star our repo, and feel free to raise issues if you have any questions.

And also wellcomes to our other works:
- [VoCo](https://github.com/Luffy03/VoCo): pretrained on the largest 3D medical image datasets (160k+), and achieves new state-of-the-art performance on multiple 3D medical image analysis tasks (~50 tasks).

