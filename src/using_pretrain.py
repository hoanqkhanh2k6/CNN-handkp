import torch
import torch.nn as nn
from torchvision import transforms, models

class Handkp_model(nn.Module):
    def __init__(self, output_size = 63, pre_trained  = True):
        super(Handkp_model, self).__init__()
        if pre_trained:
            weights = models.EfficientNet_B2_Weights.DEFAULT
        else:
            weights = None
        self.model = models.efficientnet_b2(weights=weights)
        self.model.classifier[1] = nn.Linear(in_features=1408, out_features=output_size)
        
        #dinh nghia lai classifier
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                              nn.Linear(in_features=1408, out_features=512),
                                              nn.SiLU(),
                                              
                                              nn.Dropout(p = 0.2),
                                              nn.Linear(in_features=512, out_features=256),
                                              nn.SiLU(),
                                              
                                              nn.Linear(in_features=256, out_features=output_size),
                                              nn.Sigmoid()
                                              )     
    def forward(self,x):
        return self.model(x)
        
model = Handkp_model(pre_trained=True)
model = model.to("cuda:0")

x = torch.randn(1, 3, 180, 320).to("cuda:0")
output = model(x)
#print(output.shape)  # Expected output shape: (1, 63)
#print(output)