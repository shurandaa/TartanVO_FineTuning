import pandas as pd
import shutil
import os
import glob


def returnPt(pt_time, ground_truth):
    cul = 0
    # 读取运动数据
    motion_data = pd.read_csv(ground_truth)
    motion_timestamps = motion_data['timestamp'].tolist()  # 将时间戳转换为列表
    # 读取图片时间戳
    with open(pt_time, 'r') as f:
        image_timestamps = [line.strip() for line in f]  # 将每行的时间戳读取为列表的一个元素
    selected_indices = []  # 存储选中的图片时间戳的序号

    # 遍历每两个连续的运动数据时间戳
    for i in range(1, len(motion_timestamps) - 1):
        start = motion_timestamps[i]
        end = motion_timestamps[i + 1]

        # 在两个运动时间戳之间查找图片时间戳
        for index, timestamp in enumerate(image_timestamps):
            if start <= int(timestamp) < end:
                selected_indices.append(index)
                break  # 找到第一个就停止，如果需要找所有位于两个时间戳之间的图片，可以去掉这个break

    for i in range(len(selected_indices)):
        cul += 1
    # 输出结果
    print(cul)
    return selected_indices


def copy_selected_images(numbers_list, source_dir, target_dir):
    """
    将指定数字对应的图片从源文件夹复制到目标文件夹。

    :param numbers_list: 包含图片编号的列表。
    :param source_dir: 包含源图片的目录。
    :param target_dir: 目标目录，选中的图片将被复制到这里。
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 遍历数字列表
    for number in numbers_list:
        # 构建源文件和目标文件的路径
        source_file = os.path.join(source_dir, f"{number}.png")
        target_file = os.path.join(target_dir, f"{number}.png")

        # 检查源文件是否存在
        if os.path.exists(source_file):
            # 复制文件
            shutil.copy(source_file, target_file)
            print(f"Copied {source_file} to {target_file}")
        else:
            print(f"File {source_file} not found.")


def clear_images_in_folder(folder_path):
    # 构建图片搜索模式，这里假设图片扩展名为.png
    pattern = os.path.join(folder_path, '*.png')

    # 使用glob找到所有匹配的图片文件
    image_files = glob.glob(pattern)

    # 检查是否有图片文件
    if image_files:
        print(f"找到 {len(image_files)} 张图片，正在清空文件夹...")
        for image_file in image_files:
            os.remove(image_file)  # 删除图片文件
        print("文件夹已清空。")
    else:
        print("文件夹中没有图片，无需清空。")


if __name__ == '__main__':
    ImageDir = 'data/SubT_MRS_t1/image_left'
    timestampsDir = 'data/SubT_MRS_t1/image_left/timestamps.txt'
    gdthDir = 'data/SubT_MRS_t1/ground_truth_path.csv'
    target = 'data/targetImageFolder'
    list1 = returnPt(timestampsDir, gdthDir)
    clear_images_in_folder(target)
    copy_selected_images(list1, ImageDir, target)
    print(len(list1))
