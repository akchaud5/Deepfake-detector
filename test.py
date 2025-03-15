import os
import yaml
import argparse
import torch
from trainer.exp_tester import ExpTester

def arg_parser():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config",
                        type=str,
                        default="config/Recce.yml",
                        help="Specify the path of the configuration file to be used.")
    parser.add_argument('--display', '-d', action="store_true",
                        default=False, help='Display some images.')
    return parser.parse_args()

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    arg = arg_parser()
    config_path = arg.config

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"The configuration file path '{config_path}' is invalid or missing.")

    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    data_file_path = config['data'].get('file', '')
    if not os.path.isfile(data_file_path):
        raise FileNotFoundError(f"The dataset file path '{data_file_path}' is invalid or missing.")

    tester = ExpTester(config, stage="Test")
    tester.test(display_images=arg.display)
