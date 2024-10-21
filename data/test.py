import os

from scapy.all import rdpcap
import numpy as np
from scapy.all import rdpcap, TCP, UDP
# import cv2
from PIL import Image


# 读取并处理单个会话文件（pcap 文件）
def extract_application_layer_packets(pcap_file, max_packets=16, packet_size=256):
    packets = rdpcap(pcap_file)  # 使用 scapy 读取 pcap 文件
    app_layer_packets = []
    for packet in packets[:max_packets]:
        if packet.haslayer(TCP) or packet.hasLayer(UDP):
            payload = bytes(packet[TCP].payload if packet.haslayer(TCP) else packet[UDP].payload)  # 提取应用层的数据
            if payload:
                app_layer_packets.append(payload)
    # print(app_layer_packets)

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


def save_grayscale_image_4x4_pil(session_image, pcap_file, output_dir):
    # 提取pcap文件名，不包括扩展名
    base_filename = os.path.splitext(os.path.basename(pcap_file))[0]

    # 生成图像文件名，保持与原始文件夹结构一致
    image_filename = os.path.join(output_dir, f"{base_filename}.png")

    # session_image的形状是 (n=16, sp=16, sp=16)
    n, sp, _ = session_image.shape  # n=16, sp=16

    # 创建一个空的64x64灰度图像 (4*sp, 4*sp)
    combined_image = Image.new('L', (4 * sp, 4 * sp))  # 'L'模式表示灰度图像

    # 将每个包图像放入对应位置
    for i in range(4):
        for j in range(4):
            # 提取当前包图像
            packet_image = session_image[i * 4 + j]

            # 转换为PIL图像
            packet_image_pil = Image.fromarray(packet_image)

            # 粘贴到combined_image的正确位置
            combined_image.paste(packet_image_pil, (j * sp, i * sp))

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存为灰度图像文件
    combined_image.save(image_filename)
    print(f"Image saved as {image_filename}")


def process_all_pcap_files(root_dir, output_root_dir):
    # 遍历所有子文件夹和文件
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".pcap"):
                pcap_file = os.path.join(subdir, file)

                # 处理pcap文件，生成session图像
                session_image = process_pcap_to_session_image(pcap_file)

                # 构造图像保存路径
                relative_path = os.path.relpath(subdir, root_dir)  # 获取相对路径
                output_dir = os.path.join(output_root_dir, relative_path)  # 在新目录下保持相同结构

                # 保存图像，保持与原始文件夹结构一致
                save_grayscale_image_4x4_pil(session_image, pcap_file, output_dir)


# 设置你的根目录路径
root_dir = 'D:\\PEAN-Repetition\\PreprocessedTools\\2_Session\\AllLayers\\'
output_root_dir = 'D:\\Users\\22357\\Desktop\\Thesis\\Datasets\\ALLayers'  # 新的输出目录

# 处理所有pcap文件并保存图像
process_all_pcap_files(root_dir, output_root_dir)
#     return np.array(session_images)


# 示例调用
# session_dir = 'D:\\PEAN-Repetition\\PreprocessedTools\\2_Session\\AllLayers\\FTP-ALL\\FTP.pcap.TCP_1-1-0-50_48940_1-2-179-64_25235.pcap'  # 替换为你的实际路径
# process_pcap_to_session_image(session_dir)
# process_session_pcap(session_dir)
# session_images = process_all_session_pcaps(session_dir)

# session_images: 每个会话对应一个 16x256 的图像
