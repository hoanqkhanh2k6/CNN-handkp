import torch
import torch.nn as nn
import torchvision

torch.manual_seed(42)
#np.random.seed(42)
#random.seed(42)

class Network(nn.Module):
    def __init__(self):
        self.loss = 0
        self.time = 0
        super(Network,self).__init__() #khởi tạo constructor

    # x input shape example: (batch, 3, 180, 320)
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1,padding = 1)  # -> (batch,32,180,320)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # -> (batch,32,90,160)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1,padding = 1)  # -> (batch,64,90,160)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        #self.pool2 = nn.MaxPool2d(2, 2)  # -> (batch,64,46,80)
        
    
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1,padding = 1)  # -> (batch,128,45,80)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2,2)  # -> (batch,128,45,80)
        

        self.conv4 = nn.Conv2d(128, 256, 3, stride=1,padding = 1)  # -> (batch,256,45,80)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(256,128, 3, stride=1,padding = 1)  # -> (batch,128,45,80)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        
        
        self.convskip = nn.Conv2d(64,128, 1, stride=1,padding = 0)  # -> (batch,128,45,80)
        self.bnskip = nn.BatchNorm2d(128)
        
        self.conv6 = nn.Conv2d(128,64, 3, stride=1,padding = 1)  # -> (batch,21,45,80) đầu ra 21 cho 21 heatmap
        self.bn6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()
        
        self.out = nn.Conv2d(64,21, 1, stride=1,padding = 0)  # -> (batch,21,45,80) đầu ra 21 cho 21 heatmap
        
        

        #for z from here

        #self.bn6 = nn.BatchNorm2d(63)
        #self.relu6 = nn.ReLU()
        
        self.cov7 = nn.Conv2d(21,21, 3, stride=1,padding = 1)  # -> (batch,63,45,80) đầu ra 63 cho 21 điểm (x,y,vis)
        self.bn_7 = nn.BatchNorm2d(21)
        self.relu7 = nn.ReLU()
        self.pool7 = nn.MaxPool2d(2,2)  # -> (batch,21,22,40)
        
        self.cov8 = nn.Conv2d(21,21, 3, stride=1,padding = 1)  # -> (batch,63,22,40) đầu ra 63 cho 21 điểm (x,y,vis)
        self.pool8 = nn.AdaptiveAvgPool2d((1,1))  # -> (batch,21,1,1)
        self.sigmoid = nn.Sigmoid()

    def soft_max_heat_map(self,x):
        b, c, h, w = x.size() #(batch, channels, height, width)
       
        softmaxed = nn.functional.softmax(x.view(b, c, -1), dim=-1) #phân bố xác suất theo từng px
        
        yy = torch.linspace(0, h - 1, h).to(x.device).view(1, 1, h, 1).expand(b,c,h,w).contiguous().view(b, c, -1) #tạo lưới tọa độ y
        xx = torch.linspace(0, w - 1, w).to(x.device).view(1, 1, 1, w).expand(b, c, h, w).contiguous().view(b, c, -1) #tạo lưới tọa độ x
        
        y_norm = (softmaxed * yy).sum(dim=-1) / (h - 1)  # tọa độ của y của 21 heatmap
        x_norm = (softmaxed * xx).sum(dim=-1) / (w - 1)  # tọa độ của x của 21 heatmap
        
        return [y_norm, x_norm]
    def num_flat_features(self,x):
        size = x.size()[1:]  #all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def forward(self,x):
        
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        #z = x.clone()
        
        x = self.conv2(x)
        x = self.bn2(x) 
        x = self.relu2(x)
        #x = self.pool2(x)
    
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        #x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        
        #x = x + self.bnskip(self.convskip(x))  #skip connection
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        
        x = self.out(x)  #21 heatmap
        
        
        y_norm, x_norm = self.soft_max_heat_map(x)
        
        #z = self.bn6(x)
        #z = self.relu6(z)
        #z = self.pool6(z)
        #x = x.view(-1,self.num_flat_features(x)) #flattening the layer
        z = self.cov7(x)
        z = self.bn_7(z)
        z = self.relu7(z)
        z = self.pool7(z)
        
        
        z = self.cov8(z)
        #z = self.bn_8(z)
        #z = self.relu8(z)
        z = self.pool8(z)
        
        #z = self.cov9(z)
        #z = self.bn_9(z)
        
        #z = z.view(z.size(0), -1)  # flatten before fully connected layers
        z = self.sigmoid(z)
        
        out = torch.cat((x_norm.unsqueeze(-1), y_norm.unsqueeze(-1), z.view(z.size(0), z.size(1), -1)), dim=-1)
        return out.reshape(-1,63)
    
    def forward_test(self, x):
        print(f"Input: {x.shape}")
        
        # Forward pass qua từng layer
        x = self.conv1(x); print(f"After conv1: {x.shape}")
        x = self.relu1(x)
        x = self.pool1(x); print(f"After pool1: {x.shape}")
        
        z = x.clone()

        x = self.conv2(x); print(f"After conv2: {x.shape}")
        x = self.relu2(x)
        #x = self.pool2(x); print(f"After pool2: {x.shape}")

        x = self.conv3(x); print(f"After conv3: {x.shape}")
        x = self.relu3(x)
        #x = self.pool3(x); print(f"After pool3: {x.shape}")


        x = self.conv4(x); print(f"After conv4: {x.shape}")
        x = self.relu4(x)
        #x = self.pool4(x); print(f"After pool4: {x.shape}")

        #x = self.conv5(x); print(f"After conv5: {x.shape}")
        #x = self.relu5(x)
        #x = self.pool5(x); print(f"After pool5: {x.shape}")
        
        x = self.conv6(x); print(f"After conv6: {x.shape}")
        #z = self.bn6(x)
        #z = self.relu6(z)
        #z = self.pool6(z)
        #print(f"After bn6 and relu6: {z.shape}")
        #x = x.view(-1,self.num_flat_features(x)) #flattening the layer
        z = self.cov7(z)
        print(f"After cov7: {z.shape}")
        z = self.bn_7(z)
        z = self.relu7(z)
        z = self.pool7(z)
        
        print(f"After pool7: {z.shape}")
        z = self.cov8(z)
        print(f"After cov8: {z.shape}")
        z = self.pool8(z)
        print(f"After pool8: {z.shape}")
        z = self.cov9(z)
        print(f"After cov9: {z.shape}")
        z = self.bn_9(z)
        print(f"After bn9: {z.shape}")
        z = z.view(z.size(0), -1)  # flatten before fully connected layers
        print(f"After flatten: {z.shape}")
        z = self.fully_1(z)
        print(f"After fully_1: {z.shape}")
        z = self.drop_1(z)
        z = self.fully_2(z)
        print(f"After fully_2: {z.shape}")
        z = self.sigmoid(z)
        print(f"After sigmoid: {z.shape}")

        y_norm, x_norm = self.soft_max_heat_map(x)
        
        out  = torch.cat((x_norm.unsqueeze(-1), y_norm.unsqueeze(-1), z.view(z.size(0), z.size(1), -1)), dim=-1)
        out = out.reshape(-1,63)
        print(f"Output: {out.shape}")
        #print(out)
        
        
net = Network()
#test = torch.randn(1, 3, 180, 320)  # Example input tensor
#net.forward_test(test)  # Call the test forward method



