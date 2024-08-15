import torch.nn as nn
import torch
class Recover(nn.Module):

    def __init__(self):

        super(Recover, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(102, 256, 1, 1, bias=True),
            # nn.PixelShuffle(4)
        )
        self.Pixel1 = nn.PixelShuffle(4)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1, bias=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 512, 1, 1, bias=True),
            nn.PixelShuffle(8)
        )

    def forward(self, m):

        x = self.conv1(m)
        x = self.Pixel1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x

x = torch.randn(1,102,36,63)
net = Recover();
y = net(x)
print("hello,world")
