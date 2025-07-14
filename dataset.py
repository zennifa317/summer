import json
import os

from sklearn.preprocessing import MultiLabelBinarizer
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
            self.all_classes = json.load(f)

        self.mlb = MultiLabelBinarizer(classes=self.all_classes['classes'])

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

        label = self.mlb.fit_transform([label_name])

        if self.transform:
            image = self.transform(image)

        return image, label