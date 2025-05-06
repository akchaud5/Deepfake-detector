import os
import yaml
import argparse
import torch
from trainer.exp_mgpu_trainer import ExpMultiGpuTrainer
from trainer.utils import setup_for_distributed, cleanup

def main():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="config/Recce.yml", help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    # Load main configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Load dataset-specific configuration
    with open(config['data']['file'], 'r') as file:
        dataset_config = yaml.safe_load(file)

    # Merge dataset-specific config into main config
    config.update(dataset_config)

    # Set default values
    config.setdefault('local_rank', 0)
    config.setdefault('device', 'cpu')  # Ensure device is set if not provided

    # Initialize distributed training if necessary
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        setup_for_distributed(config['local_rank'])

    # Initialize trainer
    try:
        trainer = ExpMultiGpuTrainer(config, stage="Train")
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            cleanup()
        raise e

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cleanup()

if __name__ == "__main__":
    main()
