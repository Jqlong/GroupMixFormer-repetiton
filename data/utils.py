import os

from scapy.all import rdpcap
import numpy as np


# 读取并处理单个会话文件（pcap 文件）
def process_session_pcap(pcap_file, max_packets=16, packet_size=256):
    packets = rdpcap(pcap_file)  # 使用 scapy 读取 pcap 文件
    session_data = []

    for packet in packets[:max_packets]:
        raw_bytes = bytes(packet)  # 提取原始字节数据

        # 截取或填充每个数据包到 256 字节
        if len(raw_bytes) > packet_size:
            packet_data = raw_bytes[:packet_size]
        else:
            packet_data = raw_bytes + b'\x00' * (packet_size - len(raw_bytes))

        # 转换为整数数组（字节到整数）并重塑为 16x16 图像
        packet_image = np.frombuffer(packet_data, dtype=np.uint8).reshape((16, 16))
        session_data.append(packet_image)

    # 如果数据包不足 16 个，填充全 0 数据包
    while len(session_data) < max_packets:
        session_data.append(np.zeros((16, 16), dtype=np.uint8))

    # 将数据包图像合并为会话图像 (16 x 256)
    session_image = np.vstack(session_data)
    return session_image


# 处理整个目录的会话文件
def process_all_session_pcaps(session_dir):
    session_images = []
    for session_file in os.listdir(session_dir):
        session_path = os.path.join(session_dir, session_file)
        if session_file.endswith('.pcap'):
            session_image = process_session_pcap(session_path)
            session_images.append(session_image)

    return np.array(session_images)


# 示例调用
session_dir = '/path/to/session/files'  # 替换为你的实际路径
session_images = process_all_session_pcaps(session_dir)

# session_images: 每个会话对应一个 16x256 的图像
