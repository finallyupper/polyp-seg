import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

      
class PolypDataset(Dataset):
    def __init__(self, metadata_path='../metadata.csv', root_dir='../data/', transform=None):
        self.metadata = pd.read_csv(metadata_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data = self.metadata.iloc[idx]
        
        image_path = os.path.join(self.root_dir, data['png_image_path'])
        mask_path = os.path.join(self.root_dir, data['png_mask_path'])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.5).float()  # binary mask: [0,1]

        return image, mask


def get_data_loader(data_roots, metadata_root, batch_size, workers, mean, std):
    dataset_transforms = dict(
        train=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),  
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), #Add(Yoojin) 
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        val=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        test=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]))
    loaders = {
        split: DataLoader(
            PolypDataset(
                metadata_path=os.path.join(metadata_root, f'{split}.csv'),
                root_dir=data_roots,
                transform=dataset_transforms[split]
            ),
            batch_size=batch_size,
            shuffle=split == 'train',
            num_workers=workers
            ) 
            for split in ['train', 'val', 'test']}
    return loaders 