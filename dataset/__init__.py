from .abstract_dataset import AbstractDataset
from .faceforensics import FaceForensics
from .wild_deepfake import WildDeepfake
from .celeb_df import CelebDF  # Ensure correct module name
from .dfdc import DFDC

LOADERS = {
    "FaceForensics": FaceForensics,
    "WildDeepfake": WildDeepfake,
    "CelebDF": CelebDF,
    "DFDC": DFDC,
}

def load_dataset(name):
    if name not in LOADERS:
        raise ValueError(f"Dataset '{name}' is not supported.")
    return LOADERS[name]