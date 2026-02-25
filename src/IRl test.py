import torch
import torch.nn as nn
import time
from model_define import net
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms,models
from PIL import Image, ImageOps
import numpy as np
import dataset_class


def load_model_val(path):
    net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    net.eval()
    return net

net = load_model_val(r"CNN-handkp/src/model_save_2/ver_0.10.5.pth")

Mean = [0.3799, 0.3541, 0.3407]
Std = [0.3728, 0.3592, 0.3544]

#output = net.forward(input)
#img_path = r"dataset\hand_keypoint_dataset_26k\hand_keypoint_dataset_26k\images\train\IMG_00000077.jpg"
img_path = r"CNN-handkp\src\IRL dataset\IMG_6426.jpg"
img = Image.open(img_path)
# apply EXIF orientation if present
#img = ImageOps.exif_transpose(img)
img_ = img.copy()
img_ = img_.resize((320,180))
# Validation preprocessing without padding: center-crop to 320x180 then ToTensor
img_trans = dataset_class.ImageTransform((320,180))
input = img_trans(img_, "val")

print(img_path)
print("tensor input.shape (C,H,W):", input.shape)

# Display exactly what the model sees
disp = input.permute(1, 2, 0).cpu().numpy()
disp = (disp * 255).clip(0, 255).astype(np.uint8)
plt.imshow(disp)

input = input.unsqueeze(0)
input = input.to("cpu")
input = input.to(torch.float32)


hand_key = []

output = net.forward(input)

output = output.detach().cpu().numpy()
output = output[0]

# Scale to the displayed input size (width=320, height=180)
h, w = disp.shape[0], disp.shape[1]
plt.xlim(0, w)
# image origin is top-left; invert y-axis so plotting coordinates match image coords
plt.ylim(h, 0)
for i in range(0, len(output), 3):
    # visibility bucketing
    if output[i+2] < 0.25:
        vis_cls = 0
    elif 0.25 <= output[i+2] <= 0.5:
        vis_cls = 1
    else:
        vis_cls = 2

    # clamp normalized coords for safety
    x = max(0.0, min(1.0, float(output[i])))
    y = max(0.0, min(1.0, float(output[i+1])))
    hand_key.append([int(x * w), int(y * h), vis_cls])

#print(hand_key)
for i in range(len(hand_key)):
    if hand_key[i][2] == 0:
        plt.plot(hand_key[i][0], hand_key[i][1], 'ro')  # red for low confidence
    elif hand_key[i][2] == 1:
        plt.plot(hand_key[i][0], hand_key[i][1], 'yo')  # yellow for medium confidence
    else:
        plt.plot(hand_key[i][0], hand_key[i][1], 'go')  # green for high confidence

# Helper to draw a line between two 1-based keypoint indices when both are visible
def draw_segment(idx_a_1b, idx_b_1b, color='cyan'):
    idx_a = idx_a_1b - 1
    idx_b = idx_b_1b - 1
    if idx_a < 0 or idx_b < 0 or idx_a >= len(hand_key) or idx_b >= len(hand_key):
        return
    if hand_key[idx_a][2] != 0 and hand_key[idx_b][2] != 0:
        plt.plot([hand_key[idx_a][0], hand_key[idx_b][0]],
                 [hand_key[idx_a][1], hand_key[idx_b][1]],
                 color=color, linewidth=1.5, linestyle='-')

# Connect 1→2→3→4
for idx in range(1, 5):
    draw_segment(idx, idx + 1)

# Connect 1→5
draw_segment(1, 6)

# Connect 5→6→...→21
for idx in range(6, 9):
    draw_segment(idx, idx + 1)
draw_segment(6, 10)

for idx in range(10, 13):
    draw_segment(idx, idx + 1)
draw_segment(10,14)
for idx in range(14, 17):
    draw_segment(idx, idx + 1)
draw_segment(14, 18)
for idx in range(18, 21):
    draw_segment(idx, idx + 1)
draw_segment(1,18)


#print(hand_key)
#plt.imshow(img_.resize((1920,1080)))

plt.show()
