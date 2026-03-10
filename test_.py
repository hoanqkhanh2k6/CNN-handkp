from torchvision.models import resnet50
import torch
model = resnet50(pretrained=True)

# Lấy backbone
backbone = torch.nn.Sequential(*list(model.children())[:-2])

x = torch.randn(1, 3, 180, 320)
out = backbone(x)
print("Output shape from backbone:", out.shape)
