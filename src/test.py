import torch
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import ast
import numpy as np
import dataset_class

torch.manual_seed(42)
np.random.seed(42)
#random.seed(42)
    
with open (r"D:\VS Code\vs_code\python\A4\project - Sera CV\dataset\hand_keypoint_dataset_26k\hand_keypoint_dataset_26k\labels\train\IMG_00000001.txt","r") as f:
    lines = f.read()
    tag = lines.split(" ")

    lst = []
    lst_ = []
    for i in range(5,len(tag)):
        lst.append(round(float(tag[i]),5))

    for i in range(0,len(lst),3):
        lst[i] = lst[i]*180/320 + 40/320
        lst[i+1] = lst[i+1]

        lst_.append([int(lst[i]*320),int(lst[i+1]*180),int(lst[i+2])])

    print(len(lst_))

    print(lst_)
    
img_path = r"D:\VS Code\vs_code\python\A4\project - Sera CV\dataset\hand_keypoint_dataset_26k\hand_keypoint_dataset_26k\images\train\IMG_00000001.jpg"
img = Image.open(img_path)
pad = dataset_class.PaddingImage((180,180))
padded = pad(img,40,"val")
trans = dataset_class.ImageTransform((320,180))



for i in range(len(lst_)):
    if (lst_[i][2] == 0):
        plt.plot(lst_[i][0],lst_[i][1],'ro') #red for low confidence
    elif (lst_[i][2] == 1):
        plt.plot(lst_[i][0],lst_[i][1],'yo') #yellow for medium confidence
    else:
        plt.plot(lst_[i][0],lst_[i][1],'go') #green for high confidence
plt.imshow(padded)
plt.show()