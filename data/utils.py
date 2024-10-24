import os

from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch


class TrafficImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 图片根目录，其中每个子目录代表一个类别。
            transform (callable, optional): 可选的变换函数，对样本应用的转换（例如图像增强、归一化等）。
        """
        self.root_dir = root_dir
        self.transform = transform

        # 收集所有图像路径以及它们的对应类别
        self.image_paths = []
        self.labels = []
        self.classes = os.listdir(root_dir)  # 获取所有类别
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}  # 类别映射为索引
        # print(self.class_to_idx)

        # 遍历每个类别目录并收集图片路径及其对应的标签
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.endswith(".png"):  # 只处理PNG图像
                        img_path = os.path.join(cls_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        # 返回数据集的大小
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        根据索引返回一个样本 (图像, 标签)。
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # 加载图像
        image = Image.open(img_path)

        # 应用变换（如果有）
        if self.transform:
            image = self.transform(image)
        # print(image, label)
        return image, label

if __name__ == '__main__':
    dir = 'D:\\Users\\22357\\Desktop\Thesis\\Datasets\\ALLayers'

