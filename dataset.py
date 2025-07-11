import json
import os

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

class VTDs(Dataset):
    def __init__(self, images_path, labels_path, classes_path, transform=None):
        with open(images_path, 'r') as f:
            self.images = f.readlines()
            
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
            
        with open(classes_path, 'r') as f:
            idx_to_class = json.load(f)

        class_to_idx = {}
        for key, value in idx_to_class.items():
            if value in class_to_idx:
                raise ValueError(f"Duplicate class name '{value}' found in class mapping.")
            class_to_idx[value] = key

        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = decode_image(image_path.strip())
        
        try:
            label_name = self.labels[os.path.basename(image_path.strip())]
        except KeyError:
            raise KeyError(f"No labels found for {os.path.basename(image_path.strip())}")
        
        label = []
        for l in label_name:
            if l not in self.class_to_idx:
                raise ValueError(f"Label '{l}' not found in class mapping.")
            label.append(self.class_to_idx[l])

        if self.transform:
            image = self.transform(image)

        return image, label