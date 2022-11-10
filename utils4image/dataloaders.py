import torch 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import h5py
from .config import combined_model_base 
# import torch.nn.functional as F

# import shutil
# import os
# from collections import OrderedDict
# import torch.nn as nn

class NSDFeatureDataset(Dataset): 
    def __init__(self, level='individual', mode='train', test_subject=1, part='whole'):
        if level == 'individual':
            image_feats = np.load(f'./cocofeatures/S{test_subject}/image_features.npy')
            if part == 'whole':
                response_data = np.load(f'/home/zg243/SharedRep/data/S{test_subject}_surface32k_mean.npy', mmap_mode='r')
            elif part == 'visual':
                response_data = np.load(f'/home/zg243/SharedRep/data/S{test_subject}_visual16k_mean.npy', mmap_mode='r')
            if mode == 'test':
                responses = response_data[:1000]
                feats = image_feats[:1000]
            elif mode == 'train': 
                responses = response_data[1000:9500]
                feats = image_feats[1000:9500]
            elif mode == 'val':
                responses = response_data[9500:]
                feats = image_feats[9500:]
        self.responses = responses
        self.feats = feats
        self.n_neurons = self.responses.shape[-1]
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.responses) 
  
    def __getitem__(self, index):
        'Generates one sample of data'
        X = np.asarray(self.responses[index]).astype(np.float32)
        y = np.asarray(self.feats[index]).astype(np.float32)
        return X, y


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
class NSDImageDataset(Dataset): 
    def __init__(self, level='individual', mode='train', test_subject=1, part='whole'):
        if level == 'individual':
            image_data_set = h5py.File(f"/home/zg243/nsd/stimuli/S{test_subject}_stimuli_256.h5py", 'r')
            image_data = np.copy(image_data_set['stimuli']).astype(np.float32) / 255.
            image_data_set.close()
            image_feats = np.load(f'./cocofeatures/S{test_subject}/image_features.npy')
            if part == 'whole':
                response_data = np.load(f'/home/zg243/SharedRep/data/S{test_subject}_surface32k_mean.npy', mmap_mode='r')
            elif part == 'visual':
                response_data = np.load(f'/home/zg243/SharedRep/data/S{test_subject}_visual16k_mean.npy', mmap_mode='r')
            if mode == 'test':
                responses = response_data[:1000]
                images = image_data[:1000]
                feats = image_feats[:1000]
            elif mode == 'train': 
                responses = response_data[1000:9500]
                images = image_data[1000:9500]
                feats = image_feats[1000:9500]
            elif mode == 'val':
                responses = response_data[9500:]
                images = image_data[9500:]
                feats = image_feats[9500:]
        self.responses = responses
        self.feats = feats
        images = np.moveaxis(images, 1, -1) # (n, 3, 256, 256) -> (n, 256, 256, 3)
        self.images = images
        self.n_neurons = self.responses.shape[-1]
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.responses) 
  
    def __getitem__(self, index):
        'Generates one sample of data'
        X = np.asarray(self.responses[index]).astype(np.float32)
        y_img = np.asarray(self.images[index]).astype(np.float32)
        y_feat = np.asarray(self.feats[index]).astype(np.float32)
        y_img = preprocess(y_img)
        return X, y_img, y_feat
