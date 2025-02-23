'''
Generate caches files for the datasets.

20240315 modified by ZHUANG Jiaxin.
'''

import yaml
from monai.data import load_decathlon_datalist
import nibabel as nib
import os
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)
import argparse
import sys

sys.path.append('/jhcnas1/jiaxin/codes/project_02/Medical3DMAE_v2/')
from util.data_utils_v5 import convert_dict2list, load_yaml
from util.time import getFormattedTime


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Testing the dataset.')
    # Add an argument to the parser
    parser.add_argument('--dataset_split', type=str, choices=['1k', '+2k', '+4k', '+8k', '+70k', '+100k'],
                        help='dataset split')
    # Parse the command-line arguments
    args = parser.parse_args()
    dataset_split = args.dataset_split

    config = load_yaml('../configs/datasets.yaml')

    datadir = '/data/CT/data'
    # jsondir = '/jhcnas3/CTs/jsons/'
    jsondir = '/data/CT/data/jsons/'

    fail_cases = {}
    datalists = config['Pre-trainingDatasetPath']
    jsonlists = config['Pre-trainingDatasetJson']

    for dataset_name, jsonlist in zip(datalists, jsonlists):
        # Which split, normal.
        dataset_names = config['DatasetSplit'][dataset_split]
        if dataset_name not in dataset_names:
            continue

        #!! Designed for the NLST only.
        if dataset_name not in ['NLST_convert_v1']:
            continue

        print('-'*50)
        print('{}:{}'.format(dataset_split, dataset_name))
        fail_cases[dataset_name] = []
        jsonpath = os.path.join(jsondir, jsonlist)
        datapath = os.path.join(datadir, dataset_name)
        data_dict = load_decathlon_datalist(jsonpath, False, "training", base_dir=datapath)
        print('Loading dataset dict {} with total {} files'.format(dataset_name, len(data_dict)))
        datalist = convert_dict2list(data_dict)
        print('Starting dataset {} with total {} files'.format(dataset_name, len(datalist)))
        for index, data_path in enumerate(datalist):
            if index < 23600:
                continue
            try:
                if index % 1000 == 0:
                    print(getFormattedTime()+'=> Processing dataset: {}, {}/{}'.format(dataset_name, index, len(datalist)))
                nifti_file = nib.load(data_path)
                # Get the data array from the NIFTI file
                nifti_file.get_fdata().shape
            except Exception as e:
                fail_cases[dataset_name].append(data_path)
                print('=> ', index, e)
        print('=> fails:{}'.format(len(fail_cases[dataset_name])))

    with open('fail_cases_{}.yaml'.format(dataset_split), 'w') as f:
        yaml.dump(fail_cases, f, default_flow_style=False)
