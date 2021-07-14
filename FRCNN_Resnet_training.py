# !pip install ipywidgets --user

import os
import sys
import numpy as np 
import pandas as pd

import torch
print('torch-version = ',torch.__version__ )
from PIL import Image

from torch import nn
import torchvision
print('torchvision-version = ',torchvision.__version__)
from torch.utils.data import DataLoader
from torchvision import transforms

import models
import my_utils
from datasets import WheatDataset_training

# To select the gpu
#torch.cuda.set_device(1)

base_dir = "/raid/sahil_g_ma/wheatDetection"
#base_dir = '/workspace/wheatDetection'

sys.path.append(os.path.join(base_dir, 'detection'))
from engine import train_one_epoch, evaluate
import utils

train_df = pd.read_csv(os.path.join(base_dir, 'train', 'train.csv'))
# images at index 7 and 72 are same , similarly at 16 and 85 are same
# Also they have inappropriate labels
# hence dropping them to avoid key error in dataloader and to ensure better training
train_df = train_df.drop([7, 72, 16, 85], axis=0)

# To avoid training on images with no_box

#train_df = train_df[train_df.BoxesString != 'no_box']
train_df = train_df.reset_index(drop=True)

# For training with Pseudo Labels
# pseudo_df = pd.read_csv(os.path.join(base_dir, 'submissions', 'final_sub_resnet152fpn3_igbox_pseudo2.csv'))
# pseudo_df = pseudo_df.rename(columns={'PredString':'BoxesString'})
# train_df =  pd.concat([train_df, pseudo_df]).reset_index(drop=True)


# Checking number of GPUs available
#gpu_count = torch.cuda.device_count()
#print('GPU_count=', gpu_count)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# get the model using our helper function
model = models.FRCNN_resnet50_fpn(pre_trained=False, pretrained_backbone=False)

# For using multiple GPUs
#model= nn.DataParallel(model)

# move model to the right device
model.to(device)

# training for pseudo labels
# model.load_state_dict(torch.load(os.path.join(base_dir, 'saved_models', 'frcnn_resnet152fpn_ignore_nobox5_pseudo2.pth'), 
#                                   map_location=device))

# use our dataset and defined transformations
dataset = WheatDataset_training(train_df, base_dir)
dataset_test = WheatDataset_training(train_df, base_dir)

# split the dataset in train and test set
#debugging
#indices = [i for i in range(len(dataset))]
indices = torch.randperm(len(dataset)).tolist()
dataset_train = torch.utils.data.Subset(dataset, indices[:-100])
dataset_validation = torch.utils.data.Subset(dataset_test, indices[-100:])

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=16, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

# define training and validation data loaders
# data_loader_train = torch.utils.data.DataLoader(
#     dataset_train, batch_size=8, shuffle=True, num_workers=2,
#     collate_fn=utils.collate_fn)

data_loader_validation = torch.utils.data.DataLoader(
    dataset_validation, batch_size=16, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, 
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=6,
                                               gamma=0.1)

# let's train it for 15 epochs
num_epochs = 30

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_validation, device=device)
    print('done')

#save the model
torch.save(model.state_dict(), os.path.join(base_dir, "saved_models", "frcnn_resnet50fpn_scratch20.pth"))


# Debugging
# for idx, (data, image) in enumerate(dataset):
#    print(idx)


