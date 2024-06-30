import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

def load_data(file_path):
    """ 从CSV文件加载数据，并转换为SE(3)矩阵 """
    data = np.genfromtxt(file_path, delimiter=',')
    poses = data[:, 1:]  # 假设第一列是时间戳, 后续列是平移和四元数
    timestamps = data[:, 0]  # 第一列是时间戳
    return poses, timestamps

def pos_quats2SEs(quat_datas):
    """ 将位置和四元数数据转换为SE(3)矩阵 """
    data_len = quat_datas.shape[0]
    SEs = np.zeros((data_len, 4, 4))
    for i in range(data_len):
        SE = np.eye(4)
        SE[0:3, 3] = quat_datas[i, :3]  # 平移向量
        SE[0:3, 0:3] = R.from_quat(quat_datas[i, 3:]).as_matrix()  # 四元数到旋转矩阵
        SEs[i] = SE
    return SEs

def align_trajectories(gt_SEs, est_SEs):
    """ 对齐估计轨迹到真实轨迹的尺度 """
    gt_translations = gt_SEs[:, 0:3, 3]
    est_translations = est_SEs[:, 0:3, 3]

    # 计算尺度因子
    scale_factor = np.linalg.norm(gt_translations) / np.linalg.norm(est_translations)

    # 缩放估计轨迹
    est_SEs_scaled = est_SEs.copy()
    est_SEs_scaled[:, 0:3, 3] *= scale_factor

    return gt_SEs, est_SEs_scaled, scale_factor

def SEs2pos_quats(SEs):
    """ 将SE(3)矩阵转换为位置和四元数 """
    data_len = SEs.shape[0]
    pos_quats = np.zeros((data_len, 7))
    for i in range(data_len):
        pos_quats[i, :3] = SEs[i, 0:3, 3]  # 平移向量
        pos_quats[i, 3:] = R.from_matrix(SEs[i, 0:3, 0:3]).as_quat()  # 旋转矩阵到四元数
    return pos_quats

def save_data(file_path, timestamps, pos_quats):
    """ 保存数据到CSV文件 """
    data = np.hstack((timestamps[:, np.newaxis], pos_quats))
    np.savetxt(file_path, data, delimiter=',')

def main(gt_file, est_file, gt_output_file, est_output_file):
    # 加载数据
    gt_poses, gt_timestamps = load_data(gt_file)
    est_poses, est_timestamps = load_data(est_file)

    # 转换为SE(3)
    gt_SEs = pos_quats2SEs(gt_poses)
    est_SEs = pos_quats2SEs(est_poses)

    # 对齐尺度
    gt_SEs_aligned, est_SEs_aligned, scale_factor = align_trajectories(gt_SEs, est_SEs)

    # 转换回位置和四元数
    gt_pos_quats_aligned = SEs2pos_quats(gt_SEs_aligned)
    est_pos_quats_aligned = SEs2pos_quats(est_SEs_aligned)

    # 保存数据
    save_data(gt_output_file, gt_timestamps, gt_pos_quats_aligned)
    save_data(est_output_file, est_timestamps, est_pos_quats_aligned)

    print(f"Scale factor applied: {scale_factor}")

if __name__ == "__main__":
    main("data/eURoc_DataMH4/groundtruthData.csv", "data/eURoc_DataMH4/originalPredictedData.csv",
         "data/eURoc_DataMH4/归一化数据/groundtruthData_aligned.csv", "data/eURoc_DataMH4/归一化数据/originalPredictedData_aligned.csv")
