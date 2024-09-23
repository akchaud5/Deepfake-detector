import os
import random
from os import listdir
from os.path import join
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class CelebDF(Dataset):
    def __init__(self, cfg):
        self.root = cfg['root']
        self.split = cfg['split']
        self.transforms = self.__get_transforms(cfg.get('transforms', []))
        self.images_ids = self.__get_images_ids()
        self.categories = {0: 'Fake', 1: 'Real'}  # Added categories attribute

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

    def __get_images_ids(self, limit=None):
        try:
            celeb_real = listdir(join(self.root, 'Celeb-real'))[:limit] if limit else listdir(join(self.root, 'Celeb-real'))
            celeb_fake = listdir(join(self.root, 'Celeb-synthesis'))[:limit] if limit else listdir(join(self.root, 'Celeb-synthesis'))
            youtube_real = listdir(join(self.root, 'YouTube-real'))[:limit] if limit else listdir(join(self.root, 'YouTube-real'))
        except OSError as e:
            print(f"Error accessing directories: {e}")
            raise

        images_ids = [(os.path.join('Celeb-real', img), 1) for img in celeb_real] + \
                     [(os.path.join('Celeb-synthesis', img), 0) for img in celeb_fake] + \
                     [(os.path.join('YouTube-real', img), 1) for img in youtube_real]

        random.shuffle(images_ids)
        return images_ids

    def __len__(self):
        return len(self.images_ids)

    def __getitem__(self, idx):
        image_path, label = self.images_ids[idx]
        image = self.__load_image(join(self.root, image_path))
        if self.transforms:
            image = self.transforms(image)
        return image, label

    def __load_image(self, path):
        return Image.open(path).convert('RGB')
