import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
from torchvision.io import read_image


class JPGImageDataset(Dataset):
    def __init__(self, annotations_file, image_paths, transform=None, augment_times=1):
        self.image_paths = image_paths
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform or transforms.ToTensor()
        self.augment_times = augment_times
        
    def __len__(self):
        return len(self.img_labels) * self.augment_times  # Multiple versions of dataset
        
    def __getitem__(self, idx):
        real_idx = idx // self.augment_times  # Get the real image index
        img_path = os.path.join(self.image_paths, self.img_labels.iloc[real_idx, 0])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)  # Each call will create a different transformed version
            
        label = torch.tensor(self.img_labels.iloc[real_idx, 1])
        return image, label

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        try:
            image = read_image(img_path)
        except:
            # Fallback to PIL if read_image fails
            image = Image.open(img_path).convert('RGB')
            image = transforms.ToTensor()(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label