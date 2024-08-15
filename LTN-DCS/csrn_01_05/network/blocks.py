import torch
import torch.nn as nn

class SampleNet(nn.Module):

    def __init__(self, n_group):

        super(SampleNet, self).__init__()

        self.n_group = n_group
        #0.1-0.5采样率，用1-5个Conv2d(1, 102, 32, 32, bias=False)
        self.sampleconv = nn.ModuleList([nn.Conv2d(1, 102, 32, 32, bias=False) for _ in range(n_group)])

    def forward(self, img):
        
        measurements = []

        for idx in range(self.n_group):
            measurement = self.sampleconv[idx](img)
            measurements.append(measurement)

        return measurements

class Recover(nn.Module):

    def __init__(self):

        super(Recover, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(102, 256, 1, 1, bias=True),
            nn.PixelShuffle(4)
        )
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 1, bias=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 512, 1, 1, bias=True),
            nn.PixelShuffle(8)
        )

    def forward(self, m):

        x = self.conv1(m)
        x = self.conv2(x)
        x = self.conv3(x)

        return x

class InitNet(nn.Module):

    def __init__(self, n_group):

        super(InitNet, self).__init__()

        self.n_group = n_group

        self.conv1 = nn.ModuleList([Recover() for _ in range(n_group)])
        self.conv2 = nn.ModuleList([nn.Conv2d(16, 8, 1, 1, bias=True) for _ in range(n_group-1)])
        self.conv3 = nn.Conv2d(8, 1, 3, 1, 1, bias=True)

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(102, 256, 1, 1, bias=True),
        #     nn.PixelShuffle(4)
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(16, 512, 1, 1, bias=True),
        #     nn.PixelShuffle(8)
        # )
        # self.conv3 = nn.Conv2d(8, 1, 3, 1, 1, bias=True)

    def forward(self, measurements):

        fmaps_list = []

        for idx in range(self.n_group):
            sampled_x = measurements[idx]
            fmaps = self.conv1[idx](sampled_x)
            fmaps_list.append(fmaps)

        #渐进初始重构，逐步提高分辨率
        for idx in range(self.n_group):
            if idx > 0:
                fmaps = torch.cat((fmaps_list[idx], fmaps_list[idx - 1]), 1)
                fmaps = self.conv2[idx - 1](fmaps)
                fmaps_list[idx] = fmaps
            else:
                fmaps = fmaps_list[idx]
        init_img = self.conv3(fmaps)

        return init_img, fmaps

class DeepNet(nn.Module):

    def __init__(self, n_units, n_feat, n_recurrent):
        super(DeepNet, self).__init__()

        self.n_units = n_units
        self.n_feat = n_feat
        self.n_in = n_feat
        self.n_increase = n_feat
        self.n_recurrent = n_recurrent

        self.compressunit = CompressUnit(self.n_feat) #FCM
        self.recurrentunit = RecurrentUnit(self.n_units, self.n_feat, self.n_in, self.n_increase, self.n_recurrent)
        self.fuseunits = FuseUnit(self.n_feat)

    def forward(self, init_img, fmaps):

        fmaps = self.compressunit(fmaps)
        res_fmaps = self.recurrentunit(fmaps)
        res_img = self.fuseunits(fmaps, res_fmaps) #残差图像加初始图像
        final_img = res_img + init_img

        return final_img

class CompressUnit(nn.Module):

    def __init__(self, n_feat):
        super(CompressUnit, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(8, n_feat, 2, 2, bias=True), nn.ReLU())

    def forward(self, init_img):

        fmaps = self.conv(init_img)

        return fmaps

class ResBlock(nn.Module):

    def __init__(self, n_feat):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, fmaps):
        
        res_fmaps = self.conv1(fmaps)
        res_fmaps = self.relu(res_fmaps)
        res_fmaps = self.conv2(res_fmaps)
        fmaps = fmaps + res_fmaps
        
        return fmaps

class BasicUnit(nn.Module):

    def __init__(self, n_feat, n_recurrent):
        super(BasicUnit, self).__init__()

        self.n_feat = n_feat
        self.n_recurrent = n_recurrent

        self.resblock = ResBlock(n_feat)
        self.fuseconv = nn.Conv2d(self.n_feat * self.n_recurrent, self.n_feat, 1, 1, bias=True)

    def forward(self, fmaps):

        feat_all = []
        for _ in range(self.n_recurrent):
            fmaps = self.resblock(fmaps)
            feat_all.append(fmaps)
        feat_all = torch.cat(feat_all, dim=1)
        fmaps = self.fuseconv(feat_all)

        return fmaps

class RecurrentUnit(nn.Module):

    def __init__(self, n_units, n_feat, n_in, n_increase, n_recurrent):
        #RRFM
        super(RecurrentUnit, self).__init__()

        self.n_units = n_units
        self.n_feat = n_feat
        self.n_in = n_in
        self.n_increase = n_increase
        self.n_recurrent = n_recurrent

        self.basicunits = nn.ModuleList(
            [BasicUnit(self.n_feat, self.n_recurrent) for _ in range(self.n_units)])

    def forward(self, fmaps):

        for idx in range(self.n_units):
            fmaps = self.basicunits[idx](fmaps)

        return fmaps

class FuseUnit(nn.Module):

    def __init__(self, n_feat):
        super(FuseUnit, self).__init__()

        self.conv2 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True),
            nn.ReLU(),
            nn.PixelShuffle(2), 
            nn.Conv2d(n_feat // 4, 1, 3, 1, 1, bias=True))

    def forward(self, fmaps, res_fmaps):

        fmaps = res_fmaps + fmaps
        res_img = self.conv2(fmaps)

        return res_img

