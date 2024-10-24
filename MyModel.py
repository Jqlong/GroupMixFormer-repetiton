import torch
import torch
import torch.nn as nn
import torch.optim as optim
from models.groupmixformer import GroupMixFormer
'''模型
    使用混合注意力代替多头注意力
    使用带有残差的卷积和LSTM
'''


# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 时空特征处理模块
        self.gma = GroupMixFormer()


        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 输入是1通道灰度图像
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # 输入大小为32通道的16x16特征图
        self.fc2 = nn.Linear(128, 10)  # 输出为类别数量

    def forward(self, x):
        # print(x.shape)  torch.Size([32, 1, 64, 64])
        x = self.pool(torch.relu(self.conv1(x)))  # torch.Size([32, 16, 32, 32])
        # print(x.shape)
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)  # 展平为全连接层的输入
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
