import torch
import torch.nn as nn
from optimizer_train import load_model_to_train,load_model_to_val
import time
from model_define import net
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms,models
from PIL import Image, ImageOps
import numpy as np
import dataset_class


net = load_model_to_val(net,r"D:\VS Code\vs_code\python\A4\project - Sera CV\model_save\ver_0.23_13.pth")

Mean = [0.3799, 0.3541, 0.3407]
Std = [0.3728, 0.3592, 0.3544]

#output = net.forward(input)
#img_path = r"D:\VS Code\vs_code\python\A4\project - Sera CV\dataset\hand_keypoint_dataset_26k\hand_keypoint_dataset_26k\images\train\IMG_00000001.jpg"
img_path = r"D:\VS Code\vs_code\python\A4\project - Sera CV\IRL dataset\IMG_6428 (1) (1).jpg"
#label_path = r"D:\VS Code\vs_code\python\A4\project - Sera CV\dataset\hand_keypoint_dataset_26k\hand_keypoint_dataset_26k\labels\train\IMG_00000001.txt"
img = Image.open(img_path)
# apply EXIF orientation if present
img = ImageOps.exif_transpose(img)
img_ = img.copy()
#img_ = img_.resize((1920,1080))


pad = dataset_class.PaddingImage((180,180))
img_trans = dataset_class.ImageTransform((320,180))
img_ = pad(img, 40,"val")
#img_ = img_.resize((320,180))
#img_= img_.resize((320,180))

plt.imshow(img_.resize((1920,1080)))


#img = img.resize((1920,1080))


input = img_trans(img_,"val")
#input = img_

input = input.unsqueeze(0)
input = input.to("cuda:0")
input = input.to(torch.float32)
print("tensor input.shape (B,C,H,W):", input.shape)
#print(input)
  # (1,3,180,320)

#print(input.shape)
hand_key = []
output = net.forward(input)
#print(output)

#output = output[0]
#output = output.clone()
#points = output.view(-1, 3)      # tách thành [num_points, 3]
#points[:, [0, 1]] = points[:, [0, 1]]
#output = points.view(-1)         # chuyển lại thành [num_points * 3]
output = output.detach().cpu().numpy()
output = output[0]


plt.xlim(0,1920)
# image origin is top-left; invert y-axis so plotting coordinates match image coords
plt.ylim(1080, 0)
for i in range(0,len(output),3):
    if (output[i+2] <0.25):
        output[i+2] = 0
    elif(output[i+2] >= 0.25 and output[i+2] <= 0.5):
        output[i+2] = 1
    else:
        output[i+2] =2

    hand_key.append([int((output[i])*1920),int(output[i+1]*1080),int(output[i+2])])

print(hand_key)
for i in range(len(hand_key)):
    if (hand_key[i][2] != 0):
        if (hand_key[i][2] == 0):
            plt.plot(hand_key[i][0],hand_key[i][1],'ro') #red for low confidence
        elif (hand_key[i][2] == 1):
            plt.plot(hand_key[i][0],hand_key[i][1],'yo') #yellow for medium confidence
        else:
            plt.plot(hand_key[i][0],hand_key[i][1],'go') #green for high confidence
#print(hand_key)
#plt.imshow(img_.resize((1920,1080)))

plt.show()
