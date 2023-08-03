import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

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
        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(label_path).convert("RGB"), dtype=np.float64)
        
        if self.transform is not None:
            augmentations = self.transform(image=image, label=label)
            image = augmentations["image"]
            label = augmentations["label"]
        
        return image, label
