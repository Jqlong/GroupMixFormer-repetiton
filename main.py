from torch.utils.data import DataLoader
from torchvision import transforms

from data.utils import TrafficImageDataset

# 定义图像变换（例如，调整大小，转为张量，归一化等）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转为灰度图像
    transforms.Resize((64, 64)),  # 调整大小到64x64
    transforms.ToTensor(),  # 转为Tensor
])

# 使用自定义数据集类
data_dir = 'D:\\Users\\22357\\Desktop\Thesis\\Datasets\\ALLayers'
traffic_dataset = TrafficImageDataset(root_dir=data_dir, transform=transform)

# 创建数据加载器
train_loader = DataLoader(traffic_dataset, batch_size=32, shuffle=True)

# 检查数据集
for images, labels in train_loader:
    print(f"Batch size: {images.size()}, Labels: {labels}")
    break


