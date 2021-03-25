import os
import glob
from PIL import Image
import numpy as np
import pandas as pd 
import torch 
from torch.utils.data import Dataset
from torchvision import transforms

class NotMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir 
        self.transform = transform
        self.annotations = self._get_annotations()
        self.label = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
                "H": 7, "I": 8, "J": 9}

    def __len__(self):
        return len(self.annotations) 

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1], self.annotations.iloc[index, 0])
        image = Image.open(image_path)
        target = torch.tensor(int(self.label[self.annotations.iloc[index, 1]]))

        if self.transform:
            image = self.transform(image)
        return image, target

    def _get_annotations(self):
        df = pd.DataFrame()
        for target in os.listdir(self.root_dir):
            image_list = glob.glob(os.path.join(self.root_dir, target, '*.png'))
            df = df.append([[os.path.basename(i), os.path.dirname(i)[-1]] for i in image_list])
        return df

def get_augmentation(size=224, use_flip=True, use_color_jitter=False, use_gray_scale=False, use_normalize=False):
    resize_crop = transforms.RandomResizedCrop(size=size)
    random_flip = transforms.RandomHorizontalFlip(p=0.5)
    color_jitter = transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    ], p=0.8)
    
    gray_scale = transforms.RandomGrayscale(p=0.2)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    to_tensor = transforms.ToTensor()
    
    transforms_array = np.array([resize_crop, random_flip, color_jitter, gray_scale, to_tensor, normalize])
    transforms_mask = np.array([True, use_flip, use_color_jitter, use_gray_scale, True, use_normalize])
    
    transform = transforms.Compose(transforms_array[transforms_mask])
    
    return transform