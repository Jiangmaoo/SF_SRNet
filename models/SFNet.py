import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *

# Encoder Block
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res, mode):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel, mode) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, mode, filter=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Decoder Block
class DBlock(nn.Module):
    def __init__(self, channel, num_res, mode):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel, mode) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, mode, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )


    def forward(self, x):
        x = self.main(x)
        return x


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))
class Encoder_(nn.Module):
    def __init__(self,mode,num_res=16):
        super(Encoder_, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, mode),
            EBlock(base_channel * 2, num_res, mode),
            EBlock(base_channel * 4, num_res, mode),
        ])
        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])
        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self,x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        # 256*256
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        # 128*128
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        # 64*64
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        feature_dic = {
            'res1': res1,
            'res2': res2,
            'z': z,
            'x': x,
            'x_2': x_2,
            'x_4': x_4
        }
        return feature_dic
class Decoder_(nn.Module):
    def __init__(self,mode, num_res=16):
        super(Decoder_, self).__init__()
        base_channel = 32
        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, mode),
            DBlock(base_channel * 2, num_res, mode),
            DBlock(base_channel, num_res, mode)
        ])
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        self.FAM_ = FAM(base_channel)


    def forward(self, x, x0):
        outputs = list()
        # z=torch.cat([x['z'], x0['z']],dim=1)
        z = self.FAM1(x['z'], x0['z'])
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        # 128*128
        z = self.feat_extract[3](z)
        outputs.append(z_ + x['x_4'])

        res2 = self.FAM2(x['res2'], x0['res2'])
        # res2 = torch.cat([x['res2'] + x0['res2']], dim=1)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        # 256*256
        z = self.feat_extract[4](z)
        outputs.append(z_ + x['x_2'])

        # res1 = self.FAM2(x['res1'], x0['res1'])
        res1 = self.FAM_(x['res1'], x0['res1'])
        # res1 = torch.cat([x['res1'] + x0['res1']], dim=1)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x['x'])

        return outputs


class SFNet(nn.Module):
    def __init__(self, mode, num_res=16):
        super(SFNet, self).__init__()

        self.encoder_gt=Encoder_(mode, num_res=16)
        self.encoder_x = Encoder_(mode, num_res=16)
        self.encoder_general = Encoder_(mode, num_res=16)

        self.decoder_noise = Decoder_(mode, num_res=16)
        self.decoder_gt = Decoder_(mode, num_res=16)

    def forward(self, x, gt):
        x_feature = self.encoder_x(x)
        gt_feature = self.encoder_gt(x)
        x0_feature = self.encoder_general(x)
        gt0_feature = self.encoder_general(gt)

        output_n = self.decoder_noise(x_feature, x0_feature)
        output_gt = self.decoder_gt(gt_feature, gt0_feature)

        return output_n, output_gt

def build_net(mode):
    return SFNet(mode)

if __name__ == '__main__':
    size = (3, 3, 256, 256)
    input1 = torch.ones(size)
    input2 = torch.ones(size)

    model = build_net('train')
    output_n, output_gt = model(input1, input1)
