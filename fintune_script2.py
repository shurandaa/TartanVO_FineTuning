import numpy as np
import torch
from Network.VONet import VONet  # 模型定义
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from Datasets.tartanTrajFlowDataset2 import TrajFolderDataset
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from torchvision.transforms import Normalize
import pandas as pd
from TartanVO import TartanVO


def lie_algebra_loss(R_hat_quaternion, R_quaternion):
    # 计算旋转损失为预测和真实李代数元素之间的欧氏距离
    rotation_loss = torch.norm(R_hat_quaternion - R_quaternion, dim=-1).mean()

    return rotation_loss


def pose_loss_function(T_hat, T, R_hat, R, epsilon=1e-6):
    """.0
    参数:
    T_hat: 预测的平移向量，形状为 [batch_size, 3]
    T: 真实的平移向量，形状与 T_hat 相同
    R_hat: 预测的旋转（可以是四元数或旋转矩阵），形状为 [batch_size, 4] 或 [batch_size, 3, 3]
    R: 真实的旋转，形状与 R_hat 相同
    epsilon: 用于避免除零错误的小常数

    返回:
    归一化距离损失 L_{norm\_p}
    """
    # 归一化平移向量，并计算预测和真实平移向量之间的欧几里得距离
    T_hat_norm = T_hat / torch.max(torch.norm(T_hat, dim=1, keepdim=True), torch.tensor(epsilon).to(T_hat.device))
    T_norm = T / torch.max(torch.norm(T, dim=1, keepdim=True), torch.tensor(epsilon).to(T.device))
    translation_loss = torch.norm(T_hat_norm - T_norm, dim=1).mean()

    # 计算预测和真实旋转之间的差异
    rotation_loss = lie_algebra_loss(R_hat, R)

    # 总的损失是平移损失和旋转损失的和
    total_loss = translation_loss + rotation_loss
    return total_loss, rotation_loss, translation_loss


def main():
    # 定义预处理步骤
    transform = Compose([CropCenter((640, 448)),
                         DownscaleFlow(),
                         ToTensor()])

    train_dataset = TrajFolderDataset(
        imgfolder="data/targetImageFolder",
        posefile="data/eURoc_DataMH4/groundtruthSelected.csv",
        transform=transform,
        focalx=458.6539916992,  # 根据你的相机参数调整
        focaly=457.2959899902,
        centerx=367.2149963379,
        centery=248.3750000000
    )

    # 将数据集分为前1350个用于训练，剩余部分用于验证
    train_indices = list(range(1350))
    val_indices = list(range(1350, len(train_dataset)))

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    train_dataloader = DataLoader(
        train_subset,
        batch_size=16,  # 根据你的需求调整批量大小
        shuffle=False,  # 对于预测，通常不需要打乱数据
        num_workers=4  # 根据你的系统配置调整工作线程数
    )
    val_dataloader = DataLoader(
        val_subset,
        batch_size=16,  # 根据你的需求调整批量大小
        shuffle=False,  # 对于预测，通常不需要打乱数据
        num_workers=4  # 根据你的系统配置调整工作线程数
    )

    # load the model
    model = TartanVO('tartanvo_1914.pkl')
    # 冻结 FlowNet 部分的参数

    for param in model.vonet.flowNet.parameters():
        param.requires_grad = False

    # 确认 FlowNet 部分的参数已经被冻结
    for name, param in model.vonet.named_parameters():
        if param.requires_grad and 'flowNet' in name:
            print(f"参数 {name} 将会被更新.")
        elif not param.requires_grad and 'flowNet' in name:
            print(f"参数 {name} 已被冻结，不会被更新.")

    optimizer = torch.optim.Adam(model.vonet.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                                 amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 学习率调度器

    # 设置训练的总周期数
    num_epochs = 150

    # 用于记录训练和验证损失的列表
    training_log = []
    pose_std = torch.tensor([0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=torch.float32).cuda()
    pose_std2 = np.array([0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=np.float32)

    for epoch in range(num_epochs):
        model.vonet.train()  # 设置模型为训练模式
        total_loss = 0  # 用于累积每个epoch的损失
        max_rt_loss = 0  # 初始化旋转损失的最大值
        max_tra_loss = 0  # 初始化平移损失的最大值

        # 遍历训练数据加载器中的所有批次
        for batch_idx, batch in enumerate(train_dataloader):
            # tqdm
            # 将数据移到正确的设备上（例如，GPU）
            img1, img2, intrinsics, pose = batch['img1'].to("cuda"), batch['img2'].to("cuda"), batch['intrinsic'].to(
                "cuda"), batch['motion'].to("cuda")
            motion = batch['motion']
            input = [img1, img2, intrinsics]
            optimizer.zero_grad()  # 清零梯度

            # 执行前向传播，获取模型的输出
            _, pose_hat = model.vonet(input)
            # 从模型输出中分离平移向量和旋转向量
            pose_hat = pose_hat * pose_std
            scale = torch.norm(motion[:, :3], dim=1).to("cuda")
            trans_est = pose_hat[:, :3]
            trans_est_normalized = trans_est / torch.norm(trans_est, dim=1).view(-1, 1) * scale.view(-1, 1)

            pose_hat_corrected = torch.cat([trans_est_normalized, pose_hat[:, 3:]], dim=1)

            T_hat = pose_hat_corrected[:, :3]
            R_hat = pose_hat_corrected[:, 3:]
            # 从真实的pose中分离平移向量和旋转向量，并确保在 GPU 上创建张量
            T = pose[:, :3].to(dtype=torch.float32).to("cuda", non_blocking=True)
            T.requires_grad_()
            R = pose[:, 3:].to(dtype=torch.float32).to("cuda", non_blocking=True)
            R.requires_grad_()

            # 计算损失，这里调用了自定义的损失函数
            loss, rt_loss, tra_loss = pose_loss_function(T_hat, T, R_hat, R)
            loss.backward()  # 执行反向传播
            optimizer.step()  # 更新模型参数

            total_loss += loss.item()  # 累积损失
            if rt_loss.item() > max_rt_loss:
                max_rt_loss = rt_loss.item()
            if tra_loss.item() > max_tra_loss:
                max_tra_loss = tra_loss.item()

        # 计算这个epoch的平均损失
        avg_loss = total_loss / len(train_dataloader)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Max Rt Loss: {max_rt_loss:.4f}, Max Tra Loss: {max_tra_loss:.4f}')
        # scheduler.step()

        val_totalloss = 0
        model.vonet.eval()
        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, batch in enumerate(val_dataloader):
                img1, img2, intrinsics, pose = batch['img1'].to("cuda"), batch['img2'].to("cuda"), batch[
                    'intrinsic'].to(
                    "cuda"), batch['motion'].to("cuda")
                motion2 = batch['motion']
                input = [img1, img2, intrinsics]

                # 获取模型的输出
                _, pose_hatval = model.vonet(input)
                pose_hatval = pose_hatval.data.cpu().numpy()
                pose_hatval = pose_hatval * pose_std2  # The output is normalized during training, now scale it back
                scale = np.linalg.norm(motion2[:, :3], axis=1)
                trans_est = pose_hatval[:, :3]
                trans_est = trans_est / np.linalg.norm(trans_est, axis=1).reshape(-1, 1) * scale.reshape(-1, 1)
                pose_hatval[:, :3] = trans_est
                # 从模型输出中分离平移向量和旋转向量
                T_hat = torch.tensor(pose_hatval[:, :3], dtype=torch.float32, device="cuda")
                R_hat = torch.tensor(pose_hatval[:, 3:], dtype=torch.float32, device="cuda")
                # 从真实的pose中分离平移向量和旋转向量，并确保在 GPU 上创建张量
                T = pose[:, :3].to(dtype=torch.float32).to("cuda", non_blocking=True)
                R = pose[:, 3:].to(dtype=torch.float32).to("cuda", non_blocking=True)

                # 计算损失，这里调用了自定义的损失函数
                loss, rt_loss, tra_loss = pose_loss_function(T_hat, T, R_hat, R)
                val_totalloss += loss.item()  # 累积损失

        val_avg_loss = val_totalloss / len(val_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_avg_loss:.4f}')

        # 记录训练和验证损失
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_loss': val_avg_loss
        })

        # 每训练五次保存一次模型
        if (epoch + 1) % 1 == 0:
            save_path = f'models/第五次训练/finetuneEuroc{epoch + 1}.pkl'  # 设置模型保存路径和名称
            torch.save(model.vonet.state_dict(), save_path)
            print(f'Model saved to {save_path}')

        # 可以选择在训练完成后再保存一次模型
    torch.save(model.vonet.state_dict(), 'models/第五次训练/finetune_final.pkl')
    print('Final model saved ')

    # 保存训练和验证损失到CSV文件
    df = pd.DataFrame(training_log)
    df.to_csv('models/第五次训练/training_log.csv', index=False)


if __name__ == '__main__':
    # 调用 main 函数
    main()