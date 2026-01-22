import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms,models
import torch

def make_listdata(phase = "train"):
    rootpath = r"project - Sera CV\dataset\hand_keypoint_dataset_26k\hand_keypoint_dataset_26k\images"
    target_path = osp.join(rootpath,phase,'*.jpg')
    path_list =[]
    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list

class PaddingImage:
    def __init__(self,size):
        self.img_pad = {
            "train" : transforms.Resize(size),
            "val" : transforms.Resize(size)
        }
        
    def __call__(self, img,phase):
        resize = self.img_pad[phase](img)
        padding = (70, 0, 70, 0)
        padded = ImageOps.expand(resize, border=padding, fill=(0, 0, 0))
        return padded

def compute_mean_std(file_list):
    to_tensor = transforms.ToTensor()
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_pixels_total = 0
    for file in file_list:
        img = Image.open(file)
        img = PaddingImage((180,180)) (img, 'train')
        img = to_tensor(img)
        n_pixels = img.numel() / 3
        mean += img.mean(dim=(1, 2)) * n_pixels
        std += img.std(dim=(1, 2)) * n_pixels
        n_pixels_total += n_pixels

    mean /= n_pixels_total
    std /= n_pixels_total
    return mean, std

train_path = make_listdata('train')
mean, std = compute_mean_std(train_path)
print("Mean:", mean)
print("Std:", std)

#Mean: tensor([0.3799, 0.3541, 0.3407])
#Std: tensor([0.3728, 0.3592, 0.3544])