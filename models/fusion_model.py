import torch
from torch import nn
from models.common import reflect_conv

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.cpu()
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out).cuda()

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 7 // 2
        self.conv = nn.Conv2d(1, 1, kernel_size, 1, padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.cpu()
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool + avg_pool], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return (out * x).cuda()


def AFEM(vi_feature, ir_feature):
    sigmoid = nn.Sigmoid()
    gap = nn.AdaptiveAvgPool2d(1)
    batch_size, channels, _, _ = vi_feature.size()
    space_attention_module = SpatialAttention()
    space_attention_weights = space_attention_module(ir_feature)
    ir_feature_weighted = ir_feature * space_attention_weights

    channel_attention_module = ChannelAttention((vi_feature.shape)[1])
    channel_attention_weights = channel_attention_module(vi_feature)
    vi_feature_weighted = vi_feature * channel_attention_weights
    sub_vi_ir = vi_feature_weighted - ir_feature_weighted
    vi_ir_div = sub_vi_ir * sigmoid(gap(sub_vi_ir))
    sub_ir_vi = ir_feature_weighted - vi_feature_weighted
    ir_vi_div = sub_ir_vi * sigmoid(gap(sub_ir_vi))
    vi_feature = vi_feature_weighted + ir_vi_div
    ir_feature = ir_feature_weighted + vi_ir_div

    return vi_feature, ir_feature

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)

        self.vi_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.ir_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)

        self.vi_conv3 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.ir_conv3 = reflect_conv(in_channels=32, kernel_size=3, out_channels=32, stride=1, pad=1)

        self.vi_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.ir_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)

        self.vi_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.ir_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)

        self.conv1 = reflect_conv(in_channels=256, kernel_size=3, out_channels=256, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=384, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=192, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=96, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv5 = nn.Conv2d(in_channels=48, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, y_vi_image, ir_image):
        activate = nn.LeakyReLU()
        vi_out_1 = activate(self.vi_conv1(y_vi_image))
        ir_out_1 = activate(self.ir_conv1(ir_image))
        vi_out_2 = activate(self.vi_conv2(vi_out_1))
        ir_out_2 = activate(self.ir_conv2(ir_out_1))
        fusion_vi_1 = torch.cat((vi_out_1, vi_out_2), dim=1)
        fusion_ir_1 = torch.cat((ir_out_1, ir_out_2), dim=1)

        vi_fusion_1, ir_fusion_1 = AFEM(fusion_vi_1, fusion_ir_1)
        vi_out_3 = activate(self.vi_conv3(vi_fusion_1))
        ir_out_3 = activate(self.ir_conv3(ir_fusion_1))
        vi_fusion_2, ir_fusion_2 = AFEM(vi_out_3, ir_out_3)
        vi_out_4, ir_out_4 = activate(self.vi_conv4(vi_fusion_2)), activate(self.ir_conv4(ir_fusion_2))
        vi_fusion_3, ir_fusion_3 = AFEM(vi_out_4, ir_out_4)
        vi_out_5, ir_out_5 = activate(self.vi_conv5(vi_fusion_3)), activate(self.ir_conv5(ir_fusion_3))
        final_final = torch.cat([vi_out_5, ir_out_5], dim=1)

        d_x_1 = activate(self.conv1(final_final))
        e_d_5_1 = torch.cat((vi_out_5, d_x_1), dim=1)
        d_x_2 = activate(self.conv2(e_d_5_1))
        e_d_4_2 = torch.cat((vi_out_4, d_x_2), dim=1)
        d_x_3 = activate(self.conv3(e_d_4_2))
        e_d_3_3 = torch.cat((vi_out_3, d_x_3), dim=1)
        d_x_4 = activate(self.conv4(e_d_3_3))
        e_d_2_4 = torch.cat((vi_out_2, d_x_4), dim=1)
        fused_image = nn.Tanh()(self.conv5(e_d_2_4)) / 2 + 0.5

        return fused_image

class DAPR(nn.Module):
     def __init__(self):
        super(DAPR, self).__init__()
        self.encoder = Encoder()

     def forward(self, y_vi_image, ir_image):
        fused_image = self.encoder(y_vi_image, ir_image)

        return fused_image

def AFEM(vi_feature, ir_feature):
        sigmoid = nn.Sigmoid()
        gap = nn.AdaptiveAvgPool2d(1)
        batch_size, channels, _, _ = vi_feature.size()
        space_attention_module = SpatialAttention()
        space_attention_weights = space_attention_module(ir_feature)
        ir_feature_weighted = ir_feature * space_attention_weights
        channel_attention_module = ChannelAttention((vi_feature.shape)[1])
        channel_attention_weights = channel_attention_module(vi_feature)
        vi_feature_weighted = vi_feature * channel_attention_weights
        sub_vi_ir = vi_feature_weighted - ir_feature_weighted
        vi_ir_div = sub_vi_ir * sigmoid(gap(sub_vi_ir))
        sub_ir_vi = ir_feature_weighted - vi_feature_weighted
        ir_vi_div = sub_ir_vi * sigmoid(gap(sub_ir_vi))
        vi_feature = vi_feature_weighted + ir_vi_div
        ir_feature = ir_feature_weighted + vi_ir_div

        return vi_feature, ir_feature