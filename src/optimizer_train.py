
import torch
import torch.nn as nn
from dataset_class import get_batch,get_len_batch
from model_define import net
import time
from torchvision import transforms

torch.manual_seed(42)
#np.random.seed(42)
#random.seed(42)


start_time = time.time()

torch.cuda.empty_cache()

def load_model_to_train(net,path):
    net.load_state_dict(torch.load(path))
    return net

#net = load_model_to_train(net,r"D:\VS Code\vs_code\python\A4\project - Sera CV\model_save\ver_0.13_3.pth")

#net = load_model_to_train(net,r"D:\VS Code\vs_code\python\A4\project - Sera CV\model_save_2\ver_0.2.pth")
net = net.to("cuda:0")
torch.backends.cudnn.benchmark = True

batch_size = 16
len_train = get_len_batch(batch_size,"train")

#0:2:30 for each epoch
num_epochs = 8

def masked_mse_loss(pred, target):
    # pred, target: (batch, 63) hoặc (batch, 21, 3)
    pred = pred.view(-1, 21, 3)
    target = target.view(-1, 21, 3)
    vis_mask = (target[..., 2] > 0).unsqueeze(-1)  # (batch, 21, 1)
    # Loss cho x, y (chỉ điểm visible)
    mse_xy = (pred[..., :2] - target[..., :2]) ** 2  # (batch, 21, 2)
    masked_mse_xy = mse_xy * vis_mask
    denom_xy = vis_mask.sum() * 2 if vis_mask.sum() > 0 else 1.0
    loss_xy = masked_mse_xy.sum() / denom_xy

    # Loss cho visibility (mọi điểm)
    mse_vis = (pred[..., 2] - target[..., 2]) ** 2  # (batch, 21)
    loss_vis = mse_vis.mean()

    return loss_xy + loss_vis

def wing_loss(pred, target, w=10, epsilon=2):
    x  = pred - target
    abs_x = torch.abs(x)
    C = w - w * torch.log1p(torch.tensor(w / epsilon).to("cuda:0"))
    loss = torch.where(abs_x < w, w * torch.log1p(abs_x / epsilon), abs_x - C)
    loss_ = nn.MSELoss()
    loss_ = loss_(pred, target)
    return [loss.mean(), loss_]

criterion = masked_mse_loss
#criterion = wing_loss
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

def load_model_to_val(net, path):
    net.load_state_dict(torch.load(path))
    net.eval()
    return net

def train(net,criterion,optimizer,num_epochs):
    
    for epoch in range(num_epochs):
        train_batch_iter = get_batch(batch_size,"train")
        total_loss = 0.0
        time_epoch = time.time()
        MSE_loss = 0.0
        
        
        
        for i in range(len_train):
            
            inputs, labels = next(train_batch_iter)
            inputs = inputs.to("cuda:0")
            labels = labels.to("cuda:0")
            
            optimizer.zero_grad()

            outputs = net.forward(inputs)
            #loss, _ = criterion(outputs, labels.view(-1,63),10,2) #
            loss = criterion(outputs, labels.view(-1,63))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*batch_size
            #MSE_loss = MSE_loss + _.item()*batch_size #
            
            if (i % 300 == 0):
                print("epoch {} / {}, {} %".format(epoch+1, num_epochs, round(i/(len_train)*100,2)))
            if (epoch == num_epochs - 1 and i == (len_train -1)): 
   
                print(outputs.detach().cpu()[0])
                print(labels.view(-1,63)[0])
                
            if (epoch == 0):
                net.loss = total_loss
        
        time_1_epoch  = time.time() - time_epoch
        
        print(f"Time for 1 epoch: {time_1_epoch} seconds")
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss}')
        #print("MSE loss: "  , MSE_loss) #
    print("Training complete.")
    save_path = r"D:\VS Code\vs_code\python\A4\project - Sera CV\model_save_2\ver_0.3.pth"
    torch.save(net.state_dict(), save_path)
    torch.cuda.empty_cache()
    
        
train(net,criterion,optimizer,num_epochs)


def validation (net,criterion):
   
        start_time_val = time.time()
        val_batch_iter = get_batch(batch_size,"val")
        print("len_validation ",get_len_batch(batch_size,"val"))
        total_loss = 0.0
        for i in range(get_len_batch(batch_size,"val")):
            
            
            inputs, labels = next(val_batch_iter)
            inputs = inputs.to("cuda:0")
            labels = labels.to("cuda:0")
            with torch.no_grad():
                outputs = net.forward(inputs)
                loss = criterion(outputs, labels.view(-1,63))
                total_loss += loss.item()*batch_size
                
            if (i %150 == 0):
                print("Validation {} %".format(round(i/(get_len_batch(batch_size,"val"))*100,2)))
                print("loss", total_loss)
        print(f'Validation Loss: {total_loss}')
        time_val = time.time() - start_time_val
        print(f"Time for validation: {time_val} seconds")
        print("avg_loss-1 sample: ", total_loss / get_len_batch(batch_size,"val"))


#net = load_model_to_train(net, r"D:\VS Code\vs_code\python\A4\project - Sera CV\model_save\ver_0.14_4.pth")
#validation(net,criterion)



