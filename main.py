import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.utils import TrafficImageDataset
from MyModel import SimpleCNN
from sklearn.metrics import accuracy_score, confusion_matrix

# 定义图像变换（例如，调整大小，转为张量，归一化等）
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转为灰度图像
    transforms.Resize((48, 48)),  # 调整大小到48x48  9个数据包，每个数据包16x16
    transforms.ToTensor(),  # 转为Tensor
])

# 使用自定义数据集类
data_dir = 'D:\\Users\\22357\\Desktop\Thesis\\Datasets\\ALLayers'
dataset = TrafficImageDataset(root_dir=data_dir, transform=transform)

# 将数据集分为训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 检查数据集
# for images, labels in train_loader:
#     print(f"Batch size: {images.size()}, Labels: {labels}")
#     break

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 初始化模型、损失函数和优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epoch_list = []
acc_list = []

# 训练模型
# for epoch in range(10):  # 训练10个epoch
for epoch in tqdm(range(10), desc="Processing"):
    running_loss = 0.0
    total_preds = []
    total_labels = []
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        # 前向传播
        # print(images.shape, labels.shape)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # if idx % 300 == 299:
        #     print('[%d, %5d]' % (epoch + 1, idx + 1))
        # running_loss = 0.0

        preds = torch.argmax(outputs, dim=1)
        total_preds.extend(preds.cpu().numpy())
        total_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(total_labels, total_preds)
    epoch_list.append(epoch)
    acc_list.append(acc)

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# 可视化训练过程中的准确率变化
plt.plot(epoch_list, acc_list, label='Train Accuracy')
# plt.plot(epoch_list, val_acc_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print("Finished Training")

correct = 0
total = 0

all_pred = []
all_labels = []

with torch.no_grad():  # 在评估时不需要计算梯度
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        all_pred.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        all_pred = np.array(all_pred)
        all_labels = np.array(all_labels)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")

class_names = test_loader.classes
cm = confusion_matrix(all_pred, all_labels)
# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

