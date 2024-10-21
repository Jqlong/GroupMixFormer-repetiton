import os

from scapy.all import rdpcap
import numpy as np
from scapy.all import rdpcap, TCP, UDP
import cv2


# 读取并处理单个会话文件（pcap 文件）
def extract_application_layer_packets(pcap_file, max_packets=16, packet_size=256):
    packets = rdpcap(pcap_file)  # 使用 scapy 读取 pcap 文件
    app_layer_packets = []
    for packet in packets[:max_packets]:
        if packet.haslayer(TCP) or packet.hasLayer(UDP):
            payload = bytes(packet[TCP].payload if packet.haslayer(TCP) else packet[UDP].payload)  # 提取应用层的数据
            if payload:
                app_layer_packets.append(payload)
    print(app_layer_packets)

    return app_layer_packets


def payload_to_grayscale_image(payload, sp=16):
    payload = payload[:256].ljust(256, b'\x00')

    # 转换为整数数组
    byte_array = np.frombuffer(payload, dtype=np.uint8)

    # 转换为sp x sp（16x16）的图像
    packet_image = byte_array.reshape((sp, sp))

    return packet_image


def create_session_image(app_layer_packets, n=16, sp=16):
    session_image = []

    # 保证每个会话最多有n=16个数据包，少于部分填充空数据包
    for i in range(n):
        if i < len(app_layer_packets):
            packet_image = payload_to_grayscale_image(app_layer_packets[i], sp)
        else:
            # 如果不足n个包，填充全0的数据包
            packet_image = np.zeros((sp, sp), dtype=np.uint8)

        session_image.append(packet_image)

    # 将所有packet图像合并为一个session图像
    session_image = np.stack(session_image, axis=0)

    return session_image

def process_pcap_to_session_image(pcap_file):
    # 提取应用层数据包
    app_layer_packets = extract_application_layer_packets(pcap_file)

    # 创建会话图像
    session_image = create_session_image(app_layer_packets, n=16, sp=16)

    return session_image


def save_grayscale_image_4x4(session_image, filename):
    # 将session_image的形状从 (n=16, sp=16, sp=16) 调整为4x4排列
    n, sp, _ = session_image.shape  # n=16, sp=16 (包数量和每包的图像大小)

    # 确保有4x4的排列方式
    rows = []
    for i in range(0, n, 4):
        # 将每4个图像横向拼接
        row = np.hstack(session_image[i:i + 4])
        rows.append(row)

    # 将所有行纵向拼接成最终的图像
    combined_image = np.vstack(rows)

    # 保存为灰度图像文件
    cv2.imwrite(filename, combined_image)

# 示例流程
pcap_file = 'your_session_file.pcap'
session_image = process_pcap_to_session_image(pcap_file)

# 保存生成的4x4排列的session图像为灰度图文件
save_grayscale_image_4x4(session_image, 'session_image_4x4.png')

    # print(raw_bytes)
    # hex_representation = ' '.join(f'{byte:02x}' for byte in raw_bytes)
    #
    # print(hex_representation)

    # # 截取或填充每个数据包到 256 字节
    # if len(raw_bytes) > packet_size:
    #     packet_data = raw_bytes[:packet_size]
    # else:
    #     packet_data = raw_bytes + b'\x00' * (packet_size - len(raw_bytes))

    # 转换为整数数组（字节到整数）并重塑为 16x16 图像
    # packet_image = np.frombuffer(packet_data, dtype=np.uint8).reshape((16, 16))
    # session_data.append(packet_image)

    # 如果数据包不足 16 个，填充全 0 数据包
    # while len(session_data) < max_packets:
    #     session_data.append(np.zeros((16, 16), dtype=np.uint8))
    #
    # # 将数据包图像合并为会话图像 (16 x 256)
    # session_image = np.vstack(session_data)
    # return session_image


# 处理整个目录的会话文件
# def process_all_session_pcaps(session_dir):
#     session_images = []
#     for session_file in os.listdir(session_dir):
#         session_path = os.path.join(session_dir, session_file)
#         if session_file.endswith('.pcap'):
#             session_image = process_session_pcap(session_path)
#             session_images.append(session_image)
#
#     return np.array(session_images)


# 示例调用
session_dir = 'D:\\PEAN-Repetition\\PreprocessedTools\\2_Session\\AllLayers\\FTP-ALL\\FTP.pcap.TCP_1-1-0-50_48940_1-2-179-64_25235.pcap'  # 替换为你的实际路径
process_pcap_to_session_image(session_dir)
# process_session_pcap(session_dir)
# session_images = process_all_session_pcaps(session_dir)

# session_images: 每个会话对应一个 16x256 的图像
