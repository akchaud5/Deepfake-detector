import os
import random
from os.path import join
import csv
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

SPLIT = ["train", "val", "test"]
LABEL_MAP = {"REAL": 0, "FAKE": 1}

class DFDC(Dataset):
    """
    Deepfake Detection Challenge organized by Facebook
    """

    def __init__(self, cfg, seed=2022, transforms=None):
        # Pre-check
        if cfg['split'] not in SPLIT:
            raise ValueError(f"split should be one of {SPLIT}, but found {cfg['split']}.")
        super(DFDC, self).__init__()
        print(f"Loading data from 'DFDC' of split '{cfg['split']}'\nPlease wait patiently...")
        self.root = cfg['root']
        self.split = cfg['split']
        self.transforms = self.__get_transforms(cfg.get('transforms', []))
        self.images = []
        self.targets = []
        self.num_real = 0
        self.num_fake = 0
        self.__load_data()
        assert len(self.images) == len(self.targets), "Length of images and targets not the same!"
        print(f"Data from 'DFDC' loaded.")
        print(f"Real: {self.num_real}, Fake: {self.num_fake}.")
        print(f"Dataset contains {len(self.images)} images\n")

    def __get_transforms(self, transforms_cfg):
        transform_list = []
        for transform in transforms_cfg:
            if transform['name'] == 'Resize':
                transform_list.append(transforms.Resize((transform['params']['height'], transform['params']['width'])))
            elif transform['name'] == 'HorizontalFlip':
                transform_list.append(transforms.RandomHorizontalFlip(p=transform['params']['p']))
            elif transform['name'] == 'Normalize':
                transform_list.append(transforms.ToTensor())
                transform_list.append(transforms.Normalize(mean=transform['params']['mean'], std=transform['params']['std']))
        
        if not any(isinstance(t, transforms.ToTensor) for t in transform_list):
            transform_list.insert(0, transforms.ToTensor())
        
        return transforms.Compose(transform_list)

    def __load_data(self):
        label_path = '/content/drive/MyDrive/DFDC-img_labels.csv'  # Corrected path
        with open(label_path, encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                image_path = join(self.root, row[0])
                label = int(row[1])
                self.images.append(image_path)
                self.targets.append(label)
                if label == 0:
                    self.num_real += 1
                elif label == 1:
                    self.num_fake += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.targets[idx]
        image = self.__load_image(image_path)
        if self.transforms:
            image = self.transforms(image)
        return image, label

    def __load_image(self, path):
        return Image.open(path).convert('RGB')


if __name__ == '__main__':
    import yaml

    config_path = "../config/dataset/dfdc.yml"
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = config["test_cfg"]  # Ensure to use the correct config

    def run_dataset():
        dataset = DFDC(config)
        print(f"dataset: {len(dataset)}")
        for i, (path, target) in enumerate(dataset):
            print(f"path: {path}, target: {target}")
            if i >= 9:
                break

    def run_dataloader(display_samples=False):
        from torch.utils import data
        import matplotlib.pyplot as plt

        dataset = DFDC(config)
        dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)
        print(f"dataset: {len(dataset)}")
        for i, (images, targets) in enumerate(dataloader):
            print(f"image: {images.shape}, target: {targets}")
            if display_samples:
                plt.figure()
                img = images[0].permute([1, 2, 0]).numpy()
                plt.imshow(img)
                plt.show()
            if i >= 9:
                break

    ###########################
    # run the functions below #
    ###########################

    # run_dataset()
    run_dataloader(False)
