import numpy as np
import transformations as tf
import pandas as pd

def read_trajectory_csv(filename):
    data = pd.read_csv(filename, header=None)
    trajectory = data.values.tolist()
    return trajectory

def write_trajectory_csv(filename, trajectory):
    data = []
    for entry in trajectory:
        data.append(entry)


    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, header=False)


def relative_to_absolute_trajectory(gt_file, pre_file):
    abs_trajectory = []
    current_pose = np.eye(4)

    # 提取 groundtruth 的初始位置和旋转
    prime_position = gt_file[0][1:4]
    prime_rotation = gt_file[0][4:8]

    # 设置 current_pose 为 groundtruth 的初始位置和旋转
    current_pose[:3, 3] = prime_position
    current_pose[:3, :3] = tf.quaternion_matrix(prime_rotation)[:3, :3]

    # 将初始位置和旋转添加到绝对轨迹中
    abs_trajectory.append(gt_file[0])
    print(abs_trajectory)

    # 修改循环以从第一行开始计算运动
    for i, row in enumerate(pre_file):
        if (i + 1) < len(gt_file):
            timestamp = gt_file[i+1][0]
        rel_position = row[1:4]
        rel_orientation = row[4:8]

        # 构造相对位姿矩阵
        rel_transform = np.eye(4)
        rel_transform[:3, 3] = rel_position
        rel_transform[:3, :3] = tf.quaternion_matrix(rel_orientation)[:3, :3]

        # 计算新的当前位姿
        current_pose = np.dot(current_pose, rel_transform)
        abs_position = current_pose[:3, 3]
        abs_orientation = tf.quaternion_from_matrix(current_pose)
        line = [0, 0, 0, 0, 0, 0, 0, 0]
        line[0] = timestamp
        line[1:4] = abs_position
        line[4:8] = abs_orientation
        # 使用 np.concatenate 将时间戳、位置和姿态合并为一个数组
        abs_trajectory.append(line)

    return abs_trajectory

def main():
    # 读取groundtruth文件和预测文件
    gt_trajectory = read_trajectory_csv('data/SubT_MRS_t1/groundtruthData.csv')
    relative_transforms = read_trajectory_csv('data/SubT_MRS_t1/originalPredictedData.csv')
    print(len(gt_trajectory))
    print(len(relative_transforms))


    # 生成绝对轨迹
    abs_trajectory = relative_to_absolute_trajectory(gt_trajectory, relative_transforms)

    # 删除最后一行
    abs_trajectory = abs_trajectory[:-1]

    # 将生成的绝对轨迹写入新的文件
    write_trajectory_csv('data/SubT_MRS_t1/evo_using_predict/originalPredictedData.csv', abs_trajectory)

if __name__ == '__main__':
    # 调用 main 函数
    main()



