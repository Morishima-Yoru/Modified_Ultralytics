import torch.nn as nn

print(type(nn.LayerNorm), type(nn.GroupNorm(1, 2)) )
print(nn.LayerNorm, nn.GroupNorm(1, 2) )