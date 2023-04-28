import time
import torch
from models.EfficientVIT import efficientvit_backbone_b0, efficientvit_backbone_b2, efficientvit_backbone_b1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

img = torch.randn(1, 3, 192, 256)
vit = efficientvit_backbone_b1()
start = time.time()
out = vit(img)
print(time.time() - start)
print(out["heatmaps"].shape)
print(out["pafs"].shape)
print(count_parameters(vit))