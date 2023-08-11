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
        image_path = os.path.join(self.images_dir, self.image_filenames[index])
        label_filename = self.image_filenames[index].replace("_IRRG", "_label")
        label_path = os.path.join(self.labels_dir, label_filename)
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label







