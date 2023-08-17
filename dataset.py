import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# Define the class mapping
class_mapping = {
    (255, 255, 255): 0,  # Impervious surfaces
    (0, 0, 255): 1,      # Building
    (0, 255, 255): 2,    # Low vegetation
    (0, 255, 0): 3,      # Tree
    (255, 255, 0): 4,    # Car
    (255, 0, 0): 5      # Clutter/background
}

invert_mapping = {v: k for k, v in class_mapping.items()}

def convert_from_color(arr_3d, palette=invert_mapping):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d
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
        target_image = Image.open(target_path).convert('RGB')  # Keep as RGB for multi-class
        
        if self.transform:
            data_image = self.transform(data_image)
            target_image = self.transform(target_image)
        
        target_array = convert_from_color(np.array(target_image))
            
        return data_image, target_image








