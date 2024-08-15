import torch
import torch.nn as nn
from .blocks import SampleNet, InitNet, DeepNet
from time import time

class CSRN(nn.Module):

    def __init__(self, ratio):

        super(CSRN, self).__init__()

        self.ratio_dict = {0.1: 1, 0.2: 2, 0.3: 3, 0.4: 4, 0.5: 5}
        self.n_group = self.ratio_dict[float(ratio)]
        self.blk_size = 32
        self.n_samples = int(self.blk_size**2 * 0.1)

        self.n_recurrent = 3 #RRFM循环次数
        self.n_units = 5  #RRFM循环块数
        self.n_feat = 32  #特征通道数
        
        self.samplenet = SampleNet(self.n_group)
        self.initnet = InitNet(self.n_group)
        self.deepnet = DeepNet(self.n_units, self.n_feat, self.n_recurrent)

    def forward(self, img):
        
        measurements = self.samplenet(img) #采样网络
        init_img, fmaps = self.initnet(measurements)  #初始重构
        final_img = self.deepnet(init_img, fmaps)  #残差重构

        return [init_img, final_img]
