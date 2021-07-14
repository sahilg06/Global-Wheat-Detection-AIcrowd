import os
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

import my_utils

# Training dataset
class WheatDataset_training(Dataset):
    
    def __init__(self, df, base_dir):
        self.df = df
        self.base_dir =  base_dir
        self.image_name = df['image_name']
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    
    def __len__(self):
        return len(self.image_name)
    
    def __getitem__(self, index):
        image_name = self.df.iloc[index].image_name

        path = os.path.join(self.base_dir, 'train', 'train', f'{image_name}.png')

        #for trainng with pseudo-labels
        if not os.path.isfile(path):
            path = os.path.join(self.base_dir, 'test', f'{image_name}.png')

        image = Image.open(path)
        bboxes_str = self.df.iloc[index].BoxesString
        bboxes = bboxes_str.split(';')
        n_objects = len(bboxes)  # Number of wheat heads in the given image
        
        boxes, areas = [], []
        for bbox in bboxes:
          if(bbox!='no_box'):
            box = list(map(float,bbox.split()))
            area = my_utils.area(box)
            if(area > 200000):
                continue
            boxes.append(box)
            areas.append(area)
          else:
            n_objects = 0
        
        # Convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Get the labels. We have only one class (wheat head)
        labels = torch.ones((n_objects, ), dtype=torch.int64)
        
        areas = torch.as_tensor(areas)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((n_objects, ), dtype=torch.int64) 
        
        if(n_objects == 0):
          boxes = torch.zeros((0,4), dtype=torch.float32)
          labels =  torch.zeros(0, dtype=torch.int64)
          areas = torch.zeros(0, dtype=torch.float32)
          iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([index], dtype=torch.int64), 
            'area': areas,
            'iscrowd': iscrowd
        }
        image = self.transform(image)
        target['boxes'] = target['boxes'].float()
        return image, target

# Test dataset
class WheatDataset_test(Dataset):
    
    def __init__(self, df, base_dir):
        self.base_dir = base_dir
        self.image_ids = df['image_name']
        self.domains = df['domain']
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        domain = self.domains[index]
        image = Image.open(os.path.join(self.base_dir, 'test', f'{image_id}.png'))
        image = self.transform(image)
        return image_id, image, domain
