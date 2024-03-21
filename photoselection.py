import pandas as pd

cul=0
# 读取运动数据
motion_data = pd.read_csv('data/SubT_MRS_t1/ground_truth_path.csv')
motion_timestamps = motion_data['timestamp'].tolist()  # 将时间戳转换为列表
# 读取图片时间戳
with open('data/SubT_MRS_t1/image_left/timestamps.txt', 'r') as f:
    image_timestamps = [line.strip() for line in f]  # 将每行的时间戳读取为列表的一个元素
selected_indices = []  # 存储选中的图片时间戳的序号

# 遍历每两个连续的运动数据时间戳
for i in range(len(motion_timestamps) - 1):
    start = motion_timestamps[i]
    end = motion_timestamps[i + 1]

    # 在两个运动时间戳之间查找图片时间戳
    for index, timestamp in enumerate(image_timestamps):
        if start < int(timestamp) < end:
            selected_indices.append(index)
            break  # 找到第一个就停止，如果需要找所有位于两个时间戳之间的图片，可以去掉这个break


for i in range(len(selected_indices)):
        cul += 1
# 输出结果
print(selected_indices)
print(cul)
