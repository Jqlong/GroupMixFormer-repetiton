import os


def generate(dir, label):
    """
    用于生成图片名和对应的标签
    :param dir: 文件夹地址
    :param label: 标签索引
    :return:
    """
    files = os.listdir(dir)
    listText = open('all_list.txt', 'a')
    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = file + ' ' + str(int(label)) + '\n'
        listText.write(name)
    listText.close()



outer_path = 'D:\\Users\\22357\\Desktop\Thesis\\Datasets\\ALLayers'  # 这里是你的图片的目录

if __name__ == '__main__':
    i = 0
    folderlist = os.listdir(outer_path)  # 列举文件夹
    for folder in folderlist:
        generate(os.path.join(outer_path, folder), i)
        i += 1


# if __name__ == '__main__':
#     dir = 'D:\\Users\\22357\\Desktop\Thesis\\Datasets\\ALLayers'
#     generate(dir)
