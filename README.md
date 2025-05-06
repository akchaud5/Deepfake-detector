# Attention Network for Deepfake Detection (RECCE)

This repository provides an end-to-end PyTorch implementation of the RECCE (Reconstruction-Classification) deepfake detection model, which fuses spatial RGB features with frequency-domain analysis and guided attention.

## Features
- Xception-based encoder with white-noise augmentation
- Reconstruction-guided attention and contrastive losses
- Frequency-domain filtering via learnable FFT layers
- Cross-modality fusion (RGB ↔ frequency) with CMA blocks
- Optional graph reasoning for global context aggregation
- YAML-driven configuration for models, datasets, optimizers, schedulers
- Single- and multi-GPU training (PyTorch DDP)
- Checkpointing, TensorBoard logging, and test harness

## Requirements
- Python 3.7+
- torch >=1.7.0
- torchvision
- timm
- albumentations
- numpy
- opencv-python
- pyyaml
- scikit-learn
- matplotlib
- tqdm
- tensorboard

Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Directory Structure
```
.
├── config
│   ├── Recce.yml                 # Main experiment config
│   └── dataset
│       ├── CelebDF.yml           # CelebDF dataset config
│       └── dfdc.yml              # DFDC dataset config
├── dataset                       # Dataset modules and loaders
├── loss                          # Loss functions
├── model
│   ├── common.py                 # Custom layers and utilities
│   └── network
│       └── Recce.py              # Main RECCE model definition
├── optimizer                     # Optimizer factory
├── scheduler                     # Learning rate schedulers
├── trainer
│   ├── abstract_trainer.py
│   ├── exp_mgpu_trainer.py       # Multi-GPU training loop
│   └── exp_tester.py             # Testing/inference loop
├── train.py                      # Training entrypoint
├── test.py                       # Testing entrypoint
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Configuration
All settings are driven by YAML files under `config/`.

1. **Main config**: `config/Recce.yml`
   - `model`: architecture name and parameters
   - `config`: optimizer, scheduler, device, checkpoint_dir, log_dir, etc.
   - `data`: dataset name, batch sizes, paths to dataset config, number of workers

2. **Dataset config**: `config/dataset/*.yml`
```yaml
train_cfg:
  root: "/path/to/images"
  split: "train"
  transforms:
    - name: "Resize"
      params: {height: 299, width: 299}
    - name: "Normalize"
      params: {mean: [0.5,0.5,0.5], std: [0.5,0.5,0.5]}
test_cfg:
  root: "/path/to/images"
  split: "test"
  transforms:
    - name: "Resize"
      params: {height: 299, width: 299}
    - name: "Normalize"
      params: {mean: [0.5,0.5,0.5], std: [0.5,0.5,0.5]}
```

## Usage

### Training
**Single GPU / CPU**
```bash
python train.py --config config/Recce.yml
```

**Multi-GPU (PyTorch ≥1.9)**
```bash
torchrun --nproc_per_node=4 train.py --config config/Recce.yml
```

All checkpoints and TensorBoard logs are saved under the directories specified in `config.checkpoint_dir` and `config.log_dir`.

### Testing / Inference
Once you have a trained checkpoint (`best_model.pt`), run:
```bash
python test.py --config config/Recce.yml
```
To visualize sample predictions, add `-d`:
```bash
python test.py --config config/Recce.yml -d
```
Results and logs are written to `checkpoint_dir/Recce/<run_id>/`.

## Notes
- By default, paths are set for Google Colab—please update to your local or cloud storage paths.
- On Apple M1/M2 (macOS), set `device: "mps"` in `config/Recce.yml` for inference or small-scale testing. Full training requires CUDA GPUs.
- Make sure your dataset directories match the structure expected by the dataset configs.

## Citation
If you use this code in your research, please cite our paper:
> [Insert paper title, authors, and reference here]



