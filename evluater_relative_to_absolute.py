import numpy as np
import transformations as tf
import pandas as pd

def quaternion_to_matrix(quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])


def matrix_to_quaternion(matrix):
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]

    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qw
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S=4*qx
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S=4*qy
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S=4*qz
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    return np.array([x, y, z, w])


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
    prime_rotation_xTow = [0, 0, 0, 0]
    prime_rotation_xTow[0] = prime_rotation[3]
    prime_rotation_xTow[1:4] = prime_rotation[0:3]



    # 设置 current_pose 为 groundtruth 的初始位置和旋转
    current_pose[:3, 3] = prime_position
    current_pose[:3, :3] = quaternion_to_matrix(prime_rotation)

    # 将初始位置和旋转添加到绝对轨迹中
    abs_trajectory.append(gt_file[0])
    print(abs_trajectory)

    # 修改循环以从第一行开始计算运动
    for i, row in enumerate(pre_file):
        if (i + 1) < len(gt_file):
            timestamp = gt_file[i+1][0]
        rel_position = row[1:4]
        rel_orientation = row[4:8]
        rel_orientation_xTow = [0, 0, 0, 0]
        rel_orientation_xTow[0] = rel_orientation[3]
        rel_orientation_xTow[1:4] = rel_orientation[0:3]
        # 构造相对位姿矩阵
        rel_transform = np.eye(4)
        rel_transform[:3, 3] = rel_position
        rel_transform[:3, :3] = quaternion_to_matrix(rel_orientation_xTow)

        # 计算新的当前位姿
        current_pose = current_pose.dot(rel_transform)
        abs_position = current_pose[:3, 3]
        orientation = np.eye(3)
        orientation[:3, :3] = current_pose[:3, :3]
        abs_orientation = matrix_to_quaternion(orientation)
        line = [0, 0, 0, 0, 0, 0, 0, 0]
        line[0] = timestamp
        line[1:4] = abs_position
        line[4:8] = abs_orientation
        # 使用 np.concatenate 将时间戳、位置和姿态合并为一个数组
        abs_trajectory.append(line)

    return abs_trajectory

def main():
    # 读取groundtruth文件和预测文件
    gt_trajectory = read_trajectory_csv('data/eURoc_DataMH4/groundtruthData.csv')
    relative_transforms = read_trajectory_csv('data/eURoc_DataMH4/evaluateData.csv')
    print(len(gt_trajectory))
    print(len(relative_transforms))


    # 生成绝对轨迹
    abs_trajectory = relative_to_absolute_trajectory(gt_trajectory, relative_transforms)

    # 删除最后一行
    #abs_trajectory = abs_trajectory[:-1]

    # 将生成的绝对轨迹写入新的文件
    write_trajectory_csv('data/eURoc_DataMH4/originalPredictedData.csv', abs_trajectory)

if __name__ == '__main__':
    # 调用 main 函数
    main()



