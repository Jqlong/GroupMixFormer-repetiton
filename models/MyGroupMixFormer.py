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


class ConStem:
    pass


class PatchEmbedLayer(nn.Module):
    pass


class GMA_Stage(nn.Module):
    pass


class MyGroupMixModel(nn.Module):
    def __init__(self):
        super(MyGroupMixModel, self).__init__()

        # 将图像转换为补丁
        self.conv_stem = ConStem()

        # 将经过 ConvStem 处理的特征图进一步转换为一系列的补丁（patches）
        self.patch_embed_layers = nn.ModuleList([
            PatchEmbedLayer() for i in range(4)
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
        return x

    def forward(self, x):
        # 进行特征提取
        x = self.forward_features(x)

        # 归一化
        x = self.norm(x)

        # 对张量的第2维和第3维进行平均操作
        x = x.mean(dim=(2, 3))

        # 分类
        x = self.fc(x)
        return x


