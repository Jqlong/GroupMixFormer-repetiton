import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import MyGroupMixFormer
from models.MyGroupMixFormer import GroupMixFormerCustom

from models.groupmixformer import GroupMixFormer
'''模型
    使用混合注意力代替多头注意力
    使用带有残差的卷积和LSTM
'''

class ResAtConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResAtConv, self).__init__()

        # 基本卷积层
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

        # 注意力模块中的全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # 全连接层
        self.fc = nn.Sequential(nn.Linear(out_channels, out_channels // 4),
                                nn.Sigmoid(),
                                nn.Linear(out_channels // 4, out_channels))

        # BN和残差链接
        self.bn = nn.BatchNorm2d(out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.conv(x)

        # 通道注意力：使用全局最大池化和平均池化，输入全连接层生成权重
        avg_out = self.global_avg_pool(x).view(x.size(0), -1)
        max_out = self.global_max_pool(x).view(x.size(0), -1)

        # 合并池化结果并计算注意力权重
        avg_weights = self.fc(avg_out)
        max_weights = self.fc(max_out)
        weights = torch.sigmoid(avg_weights + max_weights).view(x.size(0), x.size(1), 1, 1)

        # 使用权重调整特征图并加上残差
        x = x * weights
        x = self.bn(x) + residual
        return F.relu(x)


class STFE(nn.Module):
    def __init__(self, in_channels, conv_channels, lstm_hidden_size):
        super(STFE, self).__init__()
        self.resatconv = ResAtConv(in_channels, conv_channels)
        # 将卷积输出通道数调整为LSTM输入大小
        self.fc = nn.Linear(768, lstm_hidden_size)
        self.bilstm = nn.LSTM(32, lstm_hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x):
        # 使用ResAtConv提取空间特征
        batch_size, _, height, width = x.shape
        # print(x.shape)  # torch.Size([8, 1, 48, 48])
        x = self.resatconv(x)  # x.shape torch.Size([8, 16, 48, 48])
        # print('x.shape', x.shape)

        # 将每行的三个包视为时间步展开以便输入LSTM
        x = x.view(batch_size, height, -1)  # torch.Size([8, 48, 768])
        # print(x.shape)
        x = self.fc(x)  # torch.Size([8, 32])
        # print(x.shape)  # torch.Size([8, 48, 32])
        x, _ = self.bilstm(x)
        return x


# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleCNN, self).__init__()

        # self.st = STFE(in_channels=in_channels, conv_channels=out_channels, lstm_hidden_size=32)

        # self.gmf = GroupMixFormerCustom(num_classes=10)

        # 分类层，将LSTM输出映射到类别
        self.classifier = nn.Linear(64, 10)

        # self.fc = nn.Linear()

        # # 时空特征处理模块
        # self.gma = GroupMixFormer()
        #
        #
        # self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 输入是1通道灰度图像
        # self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(32 * 16 * 16, 128)  # 输入大小为32通道的16x16特征图
        # self.fc2 = nn.Linear(128, 10)  # 输出为类别数量

    def forward(self, x):

        x1 = self.st(x)


        # print(x.shape)
        # print(x.shape)  torch.Size([32, 1, 64, 64])
        # x = self.pool(torch.relu(self.conv1(x)))  # torch.Size([32, 16, 32, 32])
        # # print(x.shape)
        # x = self.pool(torch.relu(self.conv2(x)))
        # x = x.view(-1, 32 * 16 * 16)  # 展平为全连接层的输入
        # x = torch.relu(self.fc1(x))
        # x = self.fc2(x)
        # 使用最后一个时间步的特征进行分类
        # print(x1.shape)

        x1 = self.gmf(x)

        # x1 = x1[:, -1, :]  # 取最后一个时间步的输出
        # print(x1.shape)
        # x1 = self.classifier(x1)
        return x1
#
print(torch.cuda.is_available())
model = SimpleCNN(1, 16)
session_image = torch.randn(8, 1, 48, 48)  # 假设batch_size=8

output = model(session_image)

print(output.shape)
