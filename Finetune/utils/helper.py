"""Jiaxin ZHUANG
Modified on Apirl 29th, 2024.
"""

import yaml


def load_config_yaml_args(config_path, args):
    """Load config file based on args option, using default settings,
    and specific data settings.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    # Default settings.
    configs = {}
    for my_dict in data['default']:
        for key, value in my_dict.items():
            configs[key] = value

    # Specific data settings.
    if args.dataset_name is not None and hasattr(args, 'dataset_name'):
        for my_dict in data[args.dataset_name]:
            for key, value in my_dict.items():
                configs[key] = value

        for key, value in configs.items():
            if hasattr(args, key) and args.__dict__[key] is not None:
                print(f'Not setting key: {key} with value {value}')
            else:
                print(f'Setting key: {key} with value {value}')
                setattr(args, key, value)
    else:
        for dk in args.datasetkey:
            for key in data.keys():
                if key.startswith(dk):
                    break
            for my_dict in data[key]:
                for key, value in my_dict.items():
                    if hasattr(args, key):
                        print(f'Having key {key} with value {value}')
                    else:
                        print(f'Setting key {key} with value {value}')
                        setattr(args, key, value)

    return args


# if __name__ == "__main__":
    # threshold_organ(torch.zeros(1,12,1))
