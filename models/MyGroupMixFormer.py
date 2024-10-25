from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from models.groupmixformer import GroupMixFormer, ConvStem


class GroupMixFormerCustom(GroupMixFormer):
    def __init__(
            self,
            patch_size=4,
            in_dim=1,  # 单通道灰度输入
            num_stages=4,
            num_classes=10,  # 根据您的分类类别设置为10
            embedding_dims=[64, 128, 256, 256],  # 适当调整以适应较小输入尺寸
            serial_depths=[2, 4, 4, 2],  # 减少深度以提高效率
            num_heads=8,
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            return_interm_layers=False,
    ):
        super(GroupMixFormerCustom, self).__init__(
            patch_size=patch_size,
            in_dim=in_dim,  # 设为1以支持灰度图
            num_stages=num_stages,
            num_classes=num_classes,
            embedding_dims=embedding_dims,
            serial_depths=serial_depths,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            return_interm_layers=return_interm_layers
        )

        # 调整conv_stem接受单通道输入
        self.conv_stem = ConvStem(in_dim=1, embedding_dims=embedding_dims[0])

        # 如果不返回中间层，调整norm层
        if not self.return_interm_layers:
            self.norm4 = nn.LayerNorm(embedding_dims[-1])
            self.head = nn.Linear(embedding_dims[-1], num_classes)

    def forward(self, x):
        # 预处理或调整x为48x48
        x = F.interpolate(x, size=(48, 48), mode='bilinear', align_corners=False)
        return super().forward(x)


# 实例化自定义模型


model = GroupMixFormerCustom(num_classes=10).to("cuda:0")
model = torch.nn.parallel.DistributedDataParallel(model.to('cuda:0'))

session_image = torch.randn(8, 1, 48, 48).to("cuda:0")  # 假设batch_size=8

output = model(session_image)

print(output.shape)
