import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model

from einops import rearrange
from functools import partial
from torch import nn, einsum


class ConStem(nn.Module):
    """将输入图像转换为特征图"""

    def __init__(self, in_dim=1, embedding_dims=16):  # 输入通道和嵌入维度
        super(ConStem, self).__init__()
        mid_dim = embedding_dims // 2  # 16/2=8

        # 第一个卷积层
        self.porj1 = nn.Conv2d(in_dim, mid_dim, kernel_size=3, stride=2, padding=1)  # 不改变尺寸大小  输入1通道，输出8通道
        self.norm1 = nn.BatchNorm2d(mid_dim)  # 标准化层，用于规范特征图的分布
        self.act1 = nn.Hardswish()  # 使用激活函数

        # 第二个卷积层
        self.porj2 = nn.Conv2d(mid_dim, embedding_dims, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(embedding_dims)
        self.act2 = nn.Hardswish()  # 激活函数

    def forward(self, x):
        x = self.act1(self.norm1(self.porj1(x)))
        x = self.act2(self.norm2(self.porj2(x)))

        # x.shape torch.Size([1, 16, 12, 12])
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.pointwise_conv(self.conv1(x))
        return x


class PatchEmbedLayer(nn.Module):
    def __init__(self, patch_size=4, in_dim=16, embedding_dims=16):  # 补丁尺寸2x2，输入通道1，嵌入维度128
        super(PatchEmbedLayer, self).__init__()

        patch_size = to_2tuple(patch_size)  # 将patch_size转换为2元组  (4, 4)
        self.patch_size = patch_size

        # 可分离卷积，用于将输入特征图分割为补丁
        self.proj = SeparableConv2d(in_dim, embedding_dims, 3, patch_size, 1)
        # 标准化层
        self.norm = nn.BatchNorm2d(embedding_dims)
        # 激活函数
        self.act = nn.Hardswish()

    def forward(self, x):
        _, _, H, W = x.shape
        out_H, out_W = H // self.patch_size[0], W // self.patch_size[1]  # 划分补丁
        x = self.act(self.norm(self.proj(x)))  # x.shape torch.Size([1, 16, 3, 3])    [1, 32, 3, 3]  [1, 64, 3, 3]  [1, 64, 3, 3]
        # print('x.shape', x.shape)
        x = x.flatten(2).transpose(1, 2)   # (b, c, h, w)->(b, h * w, c)   后续注意力机制操作
        print(out_H, out_W)
        return x, (out_H, out_W)  # 3, 3


class GMA_Stage(nn.Module):
    """由多个GMA_Block组成，按顺序处理特征"""
    def __init__(self, dim, num_heads, serial_depth,  mlp_ratio=4):
        super(GMA_Stage, self).__init__()
        self.serial_depth = serial_depth


class MyGroupMixModel(nn.Module):
    def __init__(self,
                 embedding_dims=[16, 32, 64, 64]
                 ):
        super(MyGroupMixModel, self).__init__()

        # 将图像转换为补丁
        self.conv_stem = ConStem()

        # 将经过 ConvStem 处理的特征图进一步转换为一系列的补丁（patches）
        self.patch_embed_layers = nn.ModuleList([
            PatchEmbedLayer(patch_size=4, in_dim=16, embedding_dims=embedding_dims[i]) for i in range(4)
        ])

        # 模型的基本构建块和阶段
        self.groupmixformer = nn.ModuleList([
            GMA_Stage() for i in range(4)
        ])

        # 归一化
        self.norm = nn.BatchNorm2d(num_features=0)
        # 分类
        self.fc = nn.Linear(in_features=0, out_features=10)

    def forward_features(self, x):
        """特征提取  通过卷积干、补丁嵌入层和主体部分，返回中间层的特征或最终的分类特征"""
        b, _, _, _ = x.shape  # 获取形状
        x = self.conv_stem(x)  # 将图像转换为一系列特征图 形状为 x.shape torch.Size([1, 16, 12, 12])，用于提取初步特征
        out = []  # 列表

        # 进一步处理 ConvStem 输出的补丁嵌入
        for i in range(4):  # 阶段
            x_patch, (H, W) = self.patch_embed_layers[i](x)  # [i]表示模型的第几个阶段
            x = self.groupmixformer[i](x_patch, (H, W))  # 进行特征提取，对每个补丁
            # x = x.reshape(b, H, W, -1).permute(0, 3, 1, 2)  # 重塑和转置  (B, H, W, C)->(B, C, H, W)
            out.append(x)

        return out

    def forward(self, x):
        """分类"""
        # 进行特征提取
        x = self.forward_features(x)

        # 归一化
        x = self.norm(x)

        # 对张量的第2维和第3维进行平均操作
        x = x.mean(dim=(2, 3))

        # 分类
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = 'cuda:0'
    model = MyGroupMixModel().to(device)
    inputs = torch.randn(1, 1, 48, 48).to(device)
    output = model(inputs)
    print('output', output.shape)
