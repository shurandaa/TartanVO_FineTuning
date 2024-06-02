import cv2
import numpy as np
import torch
from Network.VONet import VONet  # 模型定义
from torchvision import transforms
from torch.utils.data import DataLoader
from Datasets.tartanTrajFlowDataset2 import TrajFolderDataset
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import transforms3d
import numpy as np
import pandas as pd
import os


def convert_rotation_vector_to_quaternion(data):
    # 如果接受数据是tensor类型，将其转化为numpy数组
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    # 解析位置和旋转向量
    position = data[0, :3]
    rotation_vector = data[0, 3:6]

    final_position = position.flatten()
    final_rotation_vector = rotation_vector.flatten()


    # 计算旋转向量的模（角度）
    angle = np.linalg.norm(final_rotation_vector)

    # 如果旋转向量的模为零，意味着没有旋转，返回单位四元数
    if angle == 0:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        # 计算旋转轴
        axis = final_rotation_vector / angle
        # 使用transforms3d库将旋转轴和角度转换为四元数
        quaternion = transforms3d.quaternions.axangle2quat(axis, angle)

    # 合并位置和四元数，形成一个新的7维数组
    result = np.concatenate((final_position, quaternion))
    return result


def combine_timestamps_with_poses(timestamps_path, predicted_poses, output_path):
    with open(timestamps_path, 'r') as file:
        total_lines = len(file.readlines())

    # 读取时间戳数据
    timestamps_df = pd.read_csv(timestamps_path)
    timestamps_df = timestamps_df.iloc[:-2]
    print(len(timestamps_df))
    # 确保时间戳列存在
    if 'timestamp' not in timestamps_df.columns:
        raise ValueError("Timestamps file must contain a 'timestamp' column.")

    # 获取时间戳列
    timestamps = timestamps_df['timestamp']

    # 检查时间戳数量是否与预测姿态的数量匹配
    if len(timestamps) != len(predicted_poses):
        raise ValueError("The number of timestamps does not match the number of predicted poses.")



    # 将时间戳与预测姿态结合
    full_data = np.column_stack((timestamps, predicted_poses))

    # 生成输出文件路径，并检查是否存在相同文件名，如果存在则增加序号
    index = 1
    base_filename = "evaluateData"
    extension = ".csv"
    output_filename = f"{base_filename}{extension}"
    final_output_path = os.path.join(output_path, output_filename)
    while os.path.exists(final_output_path):
        output_filename = f"{base_filename}{index}{extension}"
        final_output_path = os.path.join(output_path, output_filename)
        index += 1

    # 保存到新的 CSV 文件
    result_df = pd.DataFrame(full_data)
    result_df.to_csv(final_output_path, index=False, header=False)  # header=False 不保存列名
    print(f"Data saved to {final_output_path}")


def main():
    # 加载和初始化模型
    model = VONet()
    # 加载权重
    state_dict = torch.load('models/tartanvo_1914.pkl', map_location=torch.device('cuda'))  # 可以指定为 'cpu' 或你的 GPU 设备

    # 检查是否使用了 DataParallel 并移除 'module.' 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 加载修改后的权重
    model.load_state_dict(new_state_dict)
    model.to("cuda")
    model.eval()

    # 定义预处理步骤
    transform = Compose([CropCenter((640, 448)), DownscaleFlow(), ToTensor()])

    dataset = TrajFolderDataset(
        imgfolder="data/targetImageFolder",
        #  posefile="data/SubT_MRS_t1/ground_truth_path.csv",
        transform=transform,
        focalx=320.0,  # 根据你的相机参数调整
        focaly=320.0,
        centerx=320.0,
        centery=240.0
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # 根据你的需求调整批量大小
        shuffle=False,  # 对于预测，通常不需要打乱数据
        num_workers=2  # 根据你的系统配置调整工作线程数
    )
    poses = []
    num = 0
    for batch in dataloader:
        # 这里的 batch 是一个字典，包含了你的数据和任何其他相关信息，比如内参矩阵和运动（如果提供了位姿文件）
        img1 = batch["img1"].to("cuda")
        img2 = batch["img2"].to("cuda")  # 例如，取出图像对
        intrinsics = batch["intrinsic"].to("cuda")  # 取出内参矩阵
        input = (img1, img2, intrinsics)
        # 运行你的模型进行预测
        _, pose = model(input)  # 根据你的模型具体实现调整
        pd_pose = convert_rotation_vector_to_quaternion(pose)
        print(f"七位数组是{pd_pose}")
        poses.append(pd_pose)

    timestamps_path = 'data/SubT_MRS_t1/ground_truth_path.csv'
    output_path = 'data/SubT_MRS_t1'
    print(len(poses))

        # 结合时间戳和预测姿态，并存储结果
    combine_timestamps_with_poses(timestamps_path, poses, output_path)


if __name__ == '__main__':

    main()
