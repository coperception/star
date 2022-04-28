'''
/************************************************************************
 MIT License
 Copyright (c) 2021 AI4CE Lab@NYU, MediaBrain Group@SJTU
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 *************************************************************************/
'''
import torch.nn.functional as F
import torch.nn as nn
import torch

# # TODO: doc string
class DetBackbone(nn.Module):
    def __init__(self, height_feat_size):
        super().__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def encode(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))

        return [x, x_1, x_2, x_3, x_4]

    def decode(self, x, x_1, x_2, x_3, x_4, batch, kd_flag=False, requires_adaptive_max_pool3d=False):
        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None)) if requires_adaptive_max_pool3d else x_2
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None)) if requires_adaptive_max_pool3d else x_1
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.adaptive_max_pool3d(x, (1, None, None)) if requires_adaptive_max_pool3d else x
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        if kd_flag:
            return [res_x, x_7, x_6, x_5]
        else:
            return [res_x]


# TODO: doc string
class VAE(nn.Module):
    def __init__(self, height_feat_size):
        super().__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # ----
        self.conv_mu_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.bn_mu_1 = nn.BatchNorm2d(128)

        self.conv_mu_2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.bn_mu_2 = nn.BatchNorm2d(64)

        self.conv_mu_3 = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0)
        self.bn_mu_3 = nn.BatchNorm2d(16)

        self.conv_mu_4 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
        self.bn_mu_4 = nn.BatchNorm2d(1)
        #-----

        # ----
        self.conv_var_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.bn_var_1 = nn.BatchNorm2d(128)

        self.conv_var_2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.bn_var_2 = nn.BatchNorm2d(64)

        self.conv_var_3 = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0)
        self.bn_var_3 = nn.BatchNorm2d(16)

        self.conv_var_4 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
        self.bn_var_4 = nn.BatchNorm2d(1)
        #-----

        # self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv_global_1 = nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0)
        self.bn_global_1 = nn.BatchNorm2d(16)

        self.conv_global_2 = nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0)
        self.bn_global_2 = nn.BatchNorm2d(64)

        self.conv_global_3 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.bn_global_3 = nn.BatchNorm2d(128)

        self.conv_global_4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.bn_global_4 = nn.BatchNorm2d(256)

        #-------------
        # self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        # self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv9_1 = nn.Conv2d(32, 13, kernel_size=1, stride=1, padding=0)
        self.bn9_1 = nn.BatchNorm2d(13)

        self.conv9_2 = nn.Conv2d(13, 13, kernel_size=1, stride=1, padding=0)
        self.bn9_2 = nn.BatchNorm2d(13)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def encode(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        # x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        # x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))

        # -- Block VAE
        x_mu_1 = F.relu(self.bn_mu_1(self.conv_mu_1(x_3)))
        x_var_1 = F.relu(self.bn_var_1(self.conv_var_1(x_3)))

        x_mu_2 = F.relu(self.bn_mu_2(self.conv_mu_2(x_mu_1)))
        x_var_2 = F.relu(self.bn_var_2(self.conv_var_2(x_var_1)))
        
        x_mu_3 = F.relu(self.bn_mu_3(self.conv_mu_3(x_mu_2)))
        x_var_3 = F.relu(self.bn_var_3(self.conv_var_3(x_var_2)))

        x_mu_4 = F.relu(self.bn_mu_4(self.conv_mu_4(x_mu_3)))
        x_var_4 = F.relu(self.bn_var_4(self.conv_var_4(x_var_3)))

        return [x, x_1, x_2, x_3, x_mu_4, x_var_4]

    def decode(self, x, x_1, x_2, x_3, x_mu_4, x_var_4, batch, kd_flag=False, requires_adaptive_max_pool3d=False):
        spatial_dim = x_mu_4.shape
        std = torch.exp(x_var_4.reshape(spatial_dim[0], -1) / 2)
        q = torch.distributions.Normal(x_mu_4.reshape(spatial_dim[0], -1), std)
        z = q.rsample()
        z = z.reshape(spatial_dim).unsqueeze(1)

        x_global_1 = F.relu(self.bn_global_1(self.conv_global_1(z)))
        x_global_2 = F.relu(self.bn_global_2(self.conv_global_2(x_global_1)))
        x_global_3 = F.relu(self.bn_global_3(self.conv_global_3(x_global_2)))
        x_global_4 = F.relu(self.bn_global_4(self.conv_global_4(x_global_3)))

        # -------------------------------- Decoder Path --------------------------------
        # x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        # x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None)) if requires_adaptive_max_pool3d else x_2
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_global_4, scale_factor=(2, 2)), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None)) if requires_adaptive_max_pool3d else x_1
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.adaptive_max_pool3d(x, (1, None, None)) if requires_adaptive_max_pool3d else x
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        x_8 = F.relu(self.bn8_2(self.conv8_2(x_8)))

        x_9 = F.relu(self.bn9_1(self.conv9_1(x_8)))
        res_x = F.relu(self.bn9_2(self.conv9_2(x_9)))

        return res_x

class Backbone(nn.Module):
    def __init__(self, height_feat_size, layer1_channel=64, layer2_channel=128, layer3_channel=256):
        super().__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # keep feature map size
        # self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # channel compress for layer 1(128x128x64)
        # ----
        self.conv_mu_1= nn.Conv2d(64, layer1_channel, kernel_size=1, stride=1, padding=0)
        self.bn_mu_1 = nn.BatchNorm2d(layer1_channel)
        #-----
        
        # channel compress for layer 2(64x64x128)
        # ----
        self.conv_mu_2 = nn.Conv2d(128, layer2_channel, kernel_size=1, stride=1, padding=0)
        self.bn_mu_2 = nn.BatchNorm2d(layer2_channel)
        #-----

        # channel compress for layer 3(32x32x256)
        # ----
        self.conv_mu_3 = nn.Conv2d(256, layer3_channel, kernel_size=1, stride=1, padding=0)
        self.bn_mu_3 = nn.BatchNorm2d(layer3_channel)
        #-----
        
        

        # ----
        # self.conv_var_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        # self.bn_var_1 = nn.BatchNorm2d(128)
        #
        # self.conv_var_2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        # self.bn_var_2 = nn.BatchNorm2d(64)
        #
        # self.conv_var_3 = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0)
        # self.bn_var_3 = nn.BatchNorm2d(16)
        #
        # self.conv_var_4 = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
        # self.bn_var_4 = nn.BatchNorm2d(1)
        #-----

        # self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)


        # channel up for layer 1(128x128)
        #-------------
        self.conv_global_1 = nn.Conv2d(layer1_channel, 64, kernel_size=1, stride=1, padding=0)
        self.bn_global_1 = nn.BatchNorm2d(64)
        #-------------
        
        # channel up for layer 2(64x64)
        #-------------
        self.conv_global_2 = nn.Conv2d(layer2_channel, 128, kernel_size=1, stride=1, padding=0)
        self.bn_global_2 = nn.BatchNorm2d(128)
        #-------------
       
        # channel up for layer 3([32x32x2] -> [32x32x256])
        #-------------
        self.conv_global_3 = nn.Conv2d(layer3_channel, 256, kernel_size=1, stride=1, padding=0)
        self.bn_global_3 = nn.BatchNorm2d(256)
        #-------------


        # self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        # self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv9_1 = nn.Conv2d(32, 13, kernel_size=1, stride=1, padding=0)
        self.bn9_1 = nn.BatchNorm2d(13)

        self.conv9_2 = nn.Conv2d(13, 13, kernel_size=1, stride=1, padding=0)
        self.bn9_2 = nn.BatchNorm2d(13)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def encode(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        # x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        # x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))
        
        # -- Block VAE for layer 1
        # down to 128x128xlayer1_channel
        x_mu_1 = F.relu(self.bn_mu_1(self.conv_mu_1(x_1)))
        
        
        # -- Block VAE for layer 2
        # down to 64x64xlayer2_channel
        x_mu_2 = F.relu(self.bn_mu_2(self.conv_mu_2(x_2)))

        # -- Block VAE for layer 3
        # down to 32x32xlayer3_channel
        x_mu_3 = F.relu(self.bn_mu_3(self.conv_mu_3(x_3)))
        

        return [x, x_mu_1, x_mu_2, x_mu_3]
    
    flag = True
    def decode(self, x, x_1, x_2, x_3, batch, requires_adaptive_max_pool3d=False):
        # ------------------Channel Recovery------------------
        if self.flag:
            print("layer1 shape:", x_1.shape)
            print("layer2 shape:", x_2.shape)
            print("layer3 shape:", x_3.shape) 
            self.flag = False
        
        # channel up for layer 1
        # from 128x128xlayer1_channel to 128x128xlayer1_channel
        x_global_1 = F.relu(self.bn_global_1(self.conv_global_1(x_1))) 
        x_1 = x_global_1
        
        # channel up for layer 2
        # from 64x64xlayer2_channel to 64x64xlayer2_channel
        x_global_2 = F.relu(self.bn_global_2(self.conv_global_2(x_2)))
        x_2 = x_global_2
        
        # channel up for layer 3
        # from 32x32xlayer3_channel to 32x32xlayer3_channel
        x_global_3 = F.relu(self.bn_global_3(self.conv_global_3(x_3)))
        x_3 = x_global_3
        
        
        # -------------------------------- Decoder Path --------------------------------
        # x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        # x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None)) if requires_adaptive_max_pool3d else x_2
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()
        # print(x_2.shape)
        # [5, 128, 64, 64]

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_3, scale_factor=(2, 2)), x_2), dim=1))))
        # x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((x_3, x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None)) if requires_adaptive_max_pool3d else x_1
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.adaptive_max_pool3d(x, (1, None, None)) if requires_adaptive_max_pool3d else x
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        x_8 = F.relu(self.bn8_2(self.conv8_2(x_8)))

        x_9 = F.relu(self.bn9_1(self.conv9_1(x_8)))
        # print(self.bn9_2(self.conv9_2(x_9)).min(), self.bn9_2(self.conv9_2(x_9)).max(), self.bn9_2(self.conv9_2(x_9)).mean())
        # res_x = F.relu(self.bn9_2(self.conv9_2(x_9)))

        res_x = F.sigmoid(self.bn9_2(self.conv9_2(x_9)))

        return res_x


class STPN_KD(DetBackbone):
    def __init__(self, height_feat_size=13):
        super().__init__(height_feat_size)

    def forward(self, x):
        batch, seq, z, h, w = x.size()
        encoded_layers = super().encode(x)
        decoded_layers = super().decode(*encoded_layers, batch, kd_flag=True, requires_adaptive_max_pool3d=True)
        return (*decoded_layers, encoded_layers[3], encoded_layers[4])


class LidarEncoder(Backbone):
    def __init__(self, height_feat_size=13, layer1_channel=64, layer2_channel=128, layer3_channel=256):
        super().__init__(height_feat_size, layer1_channel, layer2_channel, layer3_channel)

    def forward(self, x):
        return super().encode(x)


class LidarDecoder(Backbone):
    def __init__(self, height_feat_size=13, layer1_channel=64, layer2_channel=128, layer3_channel=256):
        super().__init__(height_feat_size, layer1_channel, layer2_channel, layer3_channel)

    def forward(self, feature_maps, batch):
        # feature_maps: [x, x_1, x_2, x_mu_4]
        x = feature_maps[0]
        x_1 = feature_maps[1]
        x_2 = feature_maps[2]
        x_mu_4 = feature_maps[3]
        return super().decode(x, x_1, x_2, x_mu_4, batch)


class Conv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x


'''''''''''''''''''''
Added by Yiming

'''''''''''''''''''''


class Conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            is_batchnorm=True,
    ):
        super(Conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.range(start=1, end=number_of_logits, device=input.device).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input