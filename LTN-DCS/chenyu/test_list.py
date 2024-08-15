import torch
import torch.nn as nn
z = []
x = torch.randn(16,16,96,96)
s = torch.randn(1,1,2,2)
conv = nn.Conv2d(16, 1, 2 ,2, bias=True)
xxx = conv(x)
z.append(x)
z.append(s)
y = z[-1][0][0]
t = z[-1][0][0][0]
print(z)
print(x)
print(y)
print(t)
