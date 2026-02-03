import os
import h5py
import torch
import cv2
import random

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from utils.transforms import RandomRotFlip, RandomCrop, RandomBrightnessContrast, RandomGaussianNoise, ToTensor


class build_Dataset(Dataset):
    def __init__(self, args, data_dir, split, transform=None, model="None"):
        self.args = args
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.model = model
        self.rng = random.Random()
        
        self.randcolor =  RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, prob=0.8)
        self.randblur = RandomGaussianNoise(sigma=[0.1, 1.0], apply_prob=0.2)
        self.randrotflip = RandomRotFlip()
        self.randcrop = RandomCrop(self.args.patch_size)
        self.totensor = ToTensor()

        if self.split == "train_LA":
            labeled_path = os.path.join(self.data_dir + "/train.list")
            # sample_list_labeled = os.listdir(labeled_path)
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n','') for item in self.image_list]
            self.sample_list = [self.data_dir + "/" + image_name + "/mri_norm2.h5" for image_name in self.image_list]
            print("train total {} samples".format(len(self.sample_list)))
        elif self.split == "train_Pancreas":
            labeled_path = os.path.join(self.data_dir + "/train.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n','') for item in self.image_list]
            self.sample_list = [self.data_dir + "/Pancreas_h5/" + image_name + "_norm.h5" for image_name in self.image_list]
            print("train_Pancreas total {} samples".format(len(self.sample_list)))
        elif self.split == "train_BraTS2019":
            labeled_path = os.path.join(self.data_dir + "/data/train.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.sample_list = [self.data_dir + "/data/" + image_name + ".h5" for image_name in self.image_list]
            print("train_BraTS2019 total {} samples".format(len(self.sample_list)))
        elif self.split == "train_Lung":
            labeled_path = os.path.join(self.data_dir + "/train.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.sample_list = [self.data_dir + "/lung_h5/" + image_name + ".h5" for image_name in self.image_list]
            print("train_BraTS2019 total {} samples".format(len(self.sample_list)))
        elif self.split == "train_2D_list":
            labeled_path = os.path.join(self.data_dir + "/train_slices.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n','') for item in self.image_list]
            self.sample_list = [self.data_dir + "/data/slices/" + image_name + ".h5" for image_name in self.image_list]
            print("train total {} samples".format(len(self.sample_list)))
        elif self.split == "test_2D_list":
            labeled_path = os.path.join(self.data_dir + "/test.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n','') for item in self.image_list]
            self.sample_list = [self.data_dir + "/data/" + image_name + ".h5" for image_name in self.image_list]
            print("test total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)
    
    
    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(case, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        
        if "train_2D_list" in self.split:
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image']
                label = data['mask']
            image_strong = image.copy()
            image_strong = self.randblur(self.randcolor(image_strong))
            
            image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)
            image_weak = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.uint8)).long()

            sample_weak = {'image': image_weak, 'label': label}
            sample_strong = {'image': image_strong, 'label': label}
            
            new_sample = {'weak_aug': sample_weak, "strong_aug": sample_strong}
            return new_sample
        
        elif "test_2D_list" in self.split:
            image, label = torch.tensor(image.astype('float32')), torch.tensor(label.astype('float32'))
            sample = {"image": image, "label": label}
            return sample
            
        elif self.transform:
            # sample_weak = self.transform(sample)
            if self.args.dataset == "/Pancreas":
                sample = self.randcrop(sample)
            else:
                sample = self.randcrop(self.randrotflip(sample))
            image_weak, label = sample["image"], sample["label"]
            
            image_strong = image_weak.copy()
            image_strong = self.randblur(self.randcolor(image_strong))
            
            # to tensor
            image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)
            image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
            label = torch.from_numpy(label.astype(np.uint8)).long()
            
            sample_weak = {'image': image_weak, 'label': label}
            sample_strong = {'image': image_strong, 'label': label}

            # sample['weak_aug'] = sample_weak
            # sample['strong_aug'] = sample_strong
            
            new_sample = {'weak_aug': sample_weak, "strong_aug": sample_strong}
            
            return new_sample
        

    # def __getitem__(self, idx):
    #     case = self.sample_list[idx]
    #     h5f = h5py.File(case, 'r')
    #     image = h5f['image'][:]
    #     label = h5f['label'][:]
    #     sample = {'image': image, 'label': label}
        
    #     if self.transform:
    #         sample = self.transform(sample)
    #         noise_image = self.apply_intensity_augmentation(sample['image'])
    #         sample['noise_image'] = noise_image
    #     return sample
