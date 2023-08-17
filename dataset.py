import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class MyData(Dataset):
    def __init__(self, data_root, target_root, transform=None):
        self.data_root = data_root
        self.target_root = target_root
        self.transform = transform
        self.data_filenames = os.listdir(data_root)
        self.data_filenames.sort()  # Ordina i nomi dei file
        
    def __len__(self):
        return len(self.data_filenames)
    
    def __getitem__(self, idx):
        #data_filename = self.data_filenames[idx].replace("_IRRG", "")  # Rimuovi "_IRRG" dal nome del file
        #target_filename = data_filename.replace("_label", "") 
        data_path = os.path.join(self.data_root, self.data_filenames[idx])
        target_path = os.path.join(self.target_root, self.data_filenames[idx])
        
        data_image = Image.open(data_path).convert('RGB')
        target_image = Image.open(target_path).convert('L')  # Keep as RGB for multi-class
        
        if self.transform:
            data_image = self.transform(data_image)
            target_image = self.transform(target_image)
            
        return data_image, target_image








