import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from skimage.util.shape import view_as_windows


class Data(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.images[index].replace("_IRRG.tif","_label.tif"))
        image = np.array(Image.open(img_path))
        label = np.array(Image.open(label_path).convert("RGB"), dtype=np.uint8)
        
        # Define class mappings based on RGB values
        class_mapping = {
            (255, 255, 255): 0,  # Impervious surfaces
            (0, 0, 255): 1,      # Building
            (0, 255, 255): 2,    # Low vegetation
            (0, 255, 0): 3,      # Tree
            (255, 255, 0): 4,    # Car
            (255, 0, 0): 5      # Clutter/background
        }
        
        # Convert RGB label to class indices
        label_indices = np.zeros(label.shape[:2], dtype=np.uint8)
        for rgb, class_idx in class_mapping.items():
            mask = np.all(label == np.array(rgb), axis=-1)
            label_indices[mask] = class_idx
        
        if self.transform is not None:
            augmentations = self.transform(image=image, label=label_indices)
            image = augmentations["image"]
            label_indices = augmentations["label"]
        
        return image, label_indices







