"""
Dataset class for my custom dog dataset

Files in directory are named as:
{dataset_id}_{breed_id}_{img_id}_{group}.jpg
Where group denotes train (0), validation (1), or test (2)
"""
import os
import glob

import torch
import pandas as pd

from torch.utils.data import Dataset
from torchvision.transforms import Compose
from PIL import Image


class DogDataset(Dataset):
    split_id = {
        'train': 0,
        'valid': 1,
        'test': 2
    }

    def __init__(self, root: str, split: str, transform: Compose):
        img_dir = os.path.join(root, 'dogs', '*', '*.jpg')

        df = pd.DataFrame({'path': [x for x in glob.glob(img_dir, recursive=True)]})
        df['splt'] = df['path'].str.split(os.sep).str[-1].str.split('_')
        df['breed_id'] = df['splt'].str[1]
        df['breed_id'] = df['breed_id'].astype('int')
        df['group'] = df['splt'].str[3].str[:-4]
        df['group'] = df['group'].astype('int')
        
        self.labels = df.loc[df.group == self.split_id[split], ['path', 'breed_id']]
        self.transforms = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item) -> (torch.Tensor, int):
        path, breed_id = self.labels.iloc[item]
        image = Image.open(path).convert('RGB')

        image = self.transforms(image)

        return image, breed_id
