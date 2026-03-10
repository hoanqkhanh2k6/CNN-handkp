import cv2
from model_define import net
import torch
import dataset_class
import PIL


trans = dataset_class.ImageTransform((320,180))

def load_model_val(path):
    net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    net.eval()
    return net

net = load_model_val(r"CNN-handkp/src/model_save_2/ver_0.12.4.pth")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


hand_key = []
w = 1280
h = 720

def draw_segment(idx_a_1b, idx_b_1b, color=(153, 255, 255)):
    idx_a = idx_a_1b - 1
    idx_b = idx_b_1b - 1
    if idx_a < 0 or idx_b < 0 or idx_a >= len(hand_key) or idx_b >= len(hand_key):
        return
    if hand_key[idx_a][2] != 0 and hand_key[idx_b][2] != 0:
        cv2.line(frame, (hand_key[idx_a][0], hand_key[idx_a][1]), (hand_key[idx_b][0], hand_key[idx_b][1]), color, 2)

def show_hand_connections():
    # Palm
    for idx in range(1, 5):
        draw_segment(idx, idx + 1)

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
    
j = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # ← Lật ngang khung hình
    frame_input = frame.copy()
    frame_input = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
    frame_input = PIL.Image.fromarray(frame_input)
    input = trans(frame_input, "val")
    #print("input shape:", input.shape)
    
    output = net.forward(input.unsqueeze(0).to(torch.float32))
    output = output.detach().cpu().numpy()[0]
    
    hand_key = []
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
        
    if (j == 1):
        print(hand_key)
        j = 0
    
    for i in range(len(hand_key)):
        if hand_key[i][2] == 0:
            cv2.circle(frame, (hand_key[i][0], hand_key[i][1]), 5, (0, 0, 255), -1)  # red for low confidence
        elif hand_key[i][2] == 1:
            cv2.circle(frame, (hand_key[i][0], hand_key[i][1]), 5, (0, 255, 255), -1)  # yellow for medium confidence
        else:
            cv2.circle(frame, (hand_key[i][0], hand_key[i][1]), 5, (0, 255, 0), -1)  # green for high confidence
    
    # Connect 1→2→3→4
    show_hand_connections()
    cv2.imshow("Webcam", frame)   # ← HIỆN LÊN MÀN HÌNH

    if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()

