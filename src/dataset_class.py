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

#Mean: tensor([0.3799, 0.3541, 0.3407])
#Std: tensor([0.3728, 0.3592, 0.3544])

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


pad_img_path  = [r"D:\VS Code\vs_code\python\A4\project - Sera CV\pad_img\IMG_6437.jpg",]
pad_img = Image.open(pad_img_path[0])

class PaddingImage:
    def __init__(self,size):
        self.img_pad = {
            "train" : transforms.Resize(size),
            "val" : transforms.Resize(size)
        }
        
    def __call__(self, img,pad_size, phase):
        resize = self.img_pad[phase](img)
        
        l_pad = pad_size
        r_pad = 140 - pad_size

        x_start_l = np.random.randint(0, 320-l_pad)
        crop_left = pad_img.crop((x_start_l,0,x_start_l + l_pad,180  ))

        x_start_r = np.random.randint(0, 320 - r_pad)
        crop_right = pad_img.crop((x_start_r,0,x_start_r + r_pad,180  ))

        padded_img = np.hstack([crop_left, resize, crop_right])
        padded_img = Image.fromarray(padded_img.astype(np.uint8))
        return padded_img


paddingImage = PaddingImage((180,180))

def make_listdata(phase = "train"):
    # use absolute path like get_label to avoid relative-path mistakes
    rootpath = r"D:\VS Code\vs_code\python\A4\project - Sera CV\dataset\hand_keypoint_dataset_26k\hand_keypoint_dataset_26k\images"
    target_path = osp.join(rootpath,phase,'*.jpg')
    path_list =[]
    for path in glob.glob(target_path):
        path_list.append(path)
    return path_list

def get_label(phase):
    rootpath = r"D:\VS Code\vs_code\python\A4\project - Sera CV\dataset\hand_keypoint_dataset_26k\hand_keypoint_dataset_26k\labels"

    label_list =[]
    target_path = osp.join(rootpath,phase,'*.txt')
    for path in glob.glob(target_path):
        label_list.append(path)
    return label_list

class ImageTransform:
    def __init__(self, resize):
        # torchvision transforms expect size as (height, width).
        # The codebase commonly passes (width, height) (e.g. (320,180)).
        # Normalize to (height, width) here to avoid caller-side mistakes.
        if isinstance(resize, (list, tuple)) and len(resize) == 2:
            # convert (w,h) -> (h,w)
            resize_hw = (resize[1], resize[0])
        else:
            resize_hw = resize

        self.data_transform = {
            'train': transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.15,   # thay đổi độ sáng tối đa ±30%
                    contrast=0.15 ,  # thay đổi độ tương phản ±30%
                    saturation=0.1,  # thay đổi độ bão hòa ±30%
                    hue=0.1          # thay đổi sắc độ ±0.1
                ),
                transforms.ToTensor(),
                
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop(resize_hw),
                transforms.ToTensor(),

            ])
        }
    def __call__(self, img, phase):
        # Apply the chosen transform once and return the tensor
        
        return self.data_transform[phase](img)


class Dataset(data.Dataset):
    def __init__(self, data_list,labels_list, transform=None, phase='train'):
        self.file_list = data_list
        self.labels_list = labels_list
        self.transform = transform
        self.phase = phase

        self.img_pad = PaddingImage((180,180))
        self.pad = random.randint(0,140)
        
        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
         
        if (index % 500 == 0):
            self.pad = random.randint(0,140)
            
        img = self.img_pad(img,self.pad,self.phase) #320x180
        img_trans = self.transform(img, self.phase)
        # ensure image tensor has dtype float32
        if isinstance(img_trans, torch.Tensor):
            img_trans = img_trans.float()
        
        if(self.phase == 'train'):
            label_path = self.labels_list[index]
            with open(label_path, 'r') as f:
                parts = f.read().strip().split()
                lst = []
                for i in range(5,len(parts)):
                    lst.append(round(float(parts[i]),5))
                label = []
                for i in range(0,len(lst),3):
                    lst[i]  = lst[i] * (180.0/320.0) + (self.pad/320.0)  # x
                    label.append([lst[i],lst[i+1],lst[i+2]/2])
                label = torch.tensor(label,dtype = torch.float32)

        elif (self.phase == 'val'):
            label_path = self.labels_list[index]
            with open(label_path, 'r') as f:
                parts = f.read().strip().split()
                lst = []
                for i in range(5,len(parts)):
                    lst.append(round(float(parts[i]),5))
                label = []
                for i in range(0,len(lst),3):
                    lst[i]  = lst[i] * (180.0/320.0) + (self.pad/320.0)  # x
                    label.append([lst[i],lst[i+1],lst[i+2]/2])
                label = torch.tensor(label,dtype = torch.float32)
                    
        return img_trans, label

size = (320,180)
mean = (0.3799, 0.3541, 0.3407)
std = (0.3728, 0.3592, 0.3544)


list_train  = make_listdata('train')
list_val  = make_listdata('val')
labels_train = get_label('train')
labels_val = get_label('val')

data_transform = ImageTransform(size)



def get_batch(batch_size,phase = "train"):
    if (phase == "train"):
        dataset = Dataset(data_list=list_train, labels_list=labels_train, transform=data_transform, phase='train')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return iter(dataloader)
    else:
        dataset = Dataset(data_list=list_val, labels_list=labels_val, transform=data_transform, phase='val')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return iter(dataloader)


def get_len_batch(batch_size,phase = "train"):
    if (phase == "train"):
        dataset = Dataset(data_list=list_train, labels_list=labels_train, transform=data_transform, phase='train')
        return int(dataset.__len__() / batch_size)
    else:
        dataset = Dataset(data_list=list_val, labels_list=labels_val, transform=data_transform, phase='val')
        return int(dataset.__len__() / batch_size)

#test = Image.open(r"D:\VS Code\vs_code\python\A4\project - Sera CV\dataset\hand_keypoint_dataset_26k\hand_keypoint_dataset_26k\images\train\IMG_00000001.jpg")

#test = paddingImage(test,40,"val")

#test = ImageTransform((320,180))(test,"train")

#img_np = test.permute(1, 2, 0).cpu().numpy()  # [H, W, C], [0,1]
#img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

#img_pil = Image.fromarray(img_np)
#plt.imshow(img_pil)
#plt.show()


# Quick sanity check before taking a batch
#print(test.shape)


'''
batch = get_batch(1,"train")
input , label = next(batch)
input = input.squeeze(0)
print("input shape:", input.shape)
img = input.permute(1,2,0).cpu().numpy()  # [H, W, C], [0,1]
img = (img * 255).clip(0, 255).astype(np.uint8)
img = Image.fromarray(img)

label = label.squeeze(0).cpu().numpy()
for i in range(len(label)):
    if (label[i][2] == 0):
        plt.plot(label[i][0]*320,label[i][1]*180,'ro') #red for low confidence
    elif (label[i][2] == 1):
        plt.plot(label[i][0]*320,label[i][1]*180,'yo') #yellow for medium confidence
    else:
        plt.plot(label[i][0]*320,label[i][1]*180,'go') #green for high confidence



plt.imshow(img)
plt.show()
'''