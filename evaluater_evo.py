import numpy as np
from evo.core.trajectory import PosePath3D
from evo.main_ape import ape
from evo.core.metrics import PoseRelation, Unit
from scipy.spatial.transform import Rotation as R
from evo.main_rpe import rpe


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
    return PosePath3D(poses_se3=SEs)


def calculate_ate(gt_file, est_file):
    # 加载数据
    gt_poses, gt_timestamps = load_data(gt_file)
    est_poses, est_timestamps = load_data(est_file)

    # 转换为SE(3)
    gt_traj = pos_quats2SEs(gt_poses)
    est_traj = pos_quats2SEs(est_poses)

    # 计算ATE
    ate_result = ape(gt_traj, est_traj, pose_relation=PoseRelation.full_transformation, align=True)
    print("ATE result:", ate_result)

    # 计算RPE
    rpe_result = rpe(gt_traj, est_traj, delta=1, delta_unit=Unit.frames, pose_relation=PoseRelation.full_transformation)
    print("RPE result:", rpe_result)


if __name__ == "__main__":
    calculate_ate("data/SubT_MRS_t1/groundtruthData.csv", "data/SubT_MRS_t1/evo_using_predict/evaluateData.csv")

