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

import numpy as np


def rotation_vector_to_rotation_matrix(rotation_vector):
    """将旋转向量转换为旋转矩阵"""
    theta = np.linalg.norm(rotation_vector)  # 旋转向量的模，即旋转角度
    if theta == 0:
        return np.eye(3)  # 如果角度为0，返回单位矩阵

    # 单位旋转轴
    k = rotation_vector / theta

    # 罗德里格斯公式的组件
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    I = np.eye(3)

    # 使用罗德里格斯公式计算旋转矩阵
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R


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
        flow, pose = model(input)  # 根据你的模型具体实现调整
        print(pose)
        poses.append(pose.cpu().detach().numpy())
        ld = len(poses)/4000
        print(ld)


    # 初始化轨迹和方向向量
    trajectory = np.zeros((len(poses), 3))
    directions = np.zeros((len(poses), 3))

    # 计算轨迹和方向向量
    for i, pose in enumerate(poses):
        translation = pose[0, :3]  # 使用 np.squeeze() 去除单维度
        rotation = pose[0, 3:]

        # 累积平移以构建轨迹
        if i == 0:
            trajectory[i] = translation
        else:
            trajectory[i] = trajectory[i - 1] + translation

        # 将旋转向量转换为旋转矩阵
        R = rotation_vector_to_rotation_matrix(rotation)

        # 应用旋转矩阵到初始方向向量 (这里我们选择 z 轴方向的单位向量)
        direction = R @ np.array([0, 0, 1])
        directions[i] = direction

    # 绘制轨迹
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], marker='o')

    # 在每个点上绘制方向
    scale_factor = 0.1  # 用于调整方向向量的长度
    for i in range(len(poses)):
        ax.quiver(trajectory[i, 0], trajectory[i, 1], trajectory[i, 2],
                  directions[i, 0], directions[i, 1], directions[i, 2],
                  length=scale_factor, normalize=True, color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    # 处理你的预测结果...


if __name__ == '__main__':
    # 调用 main 函数
    main()
