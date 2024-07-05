import numpy as np
import torch
from Network.VONet import VONet  # 模型定义
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from Datasets.tartanTrajFlowDataset2 import TrajFolderDataset
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from torchvision.transforms import Normalize

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
    transform = Compose([CropCenter((640, 448)), DownscaleFlow(), ToTensor(),
                         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

    train_dataset = TrajFolderDataset(
        imgfolder="data/targetImageFolder",
        posefile="data/SubT_MRS_t1/ground_truth_path.csv",
        transform=transform,
        focalx=320.0,  # 根据你的相机参数调整
        focaly=320.0,
        centerx=320.0,
        centery=240.0
    )

    subset_indices = list(range(500))
    train_subset = Subset(train_dataset, subset_indices)

    train_dataloader = DataLoader(
        train_subset,
        batch_size=1,  # 根据你的需求调整批量大小
        shuffle=False,  # 对于预测，通常不需要打乱数据
        num_workers=2  # 根据你的系统配置调整工作线程数
    )

    #load the model
    model = VONet()

    #load etc
    state_dict = torch.load('models/tartanvo_1914.pkl', map_location=torch.device('cuda'))

    # 检查是否使用了 DataParallel 并移除 'module.' 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 加载修改后的权重
    model.load_state_dict(new_state_dict)
    model.to("cuda")
    # 冻结 FlowNet 部分的参数
    for param in model.flowNet.parameters():
        param.requires_grad = False

    # 确认 FlowNet 部分的参数已经被冻结
    for name, param in model.named_parameters():
        if param.requires_grad and 'flowNet' in name:
            print(f"参数 {name} 将会被更新.")
        elif not param.requires_grad and 'flowNet' in name:
            print(f"参数 {name} 已被冻结，不会被更新.")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)

    # 设置训练的总周期数
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        total_loss = 0  # 用于累积每个epoch的损失
        max_rt_loss = 0  # 初始化旋转损失的最大值
        max_tra_loss = 0  # 初始化平移损失的最大值

        # 遍历训练数据加载器中的所有批次
        for batch_idx, batch in enumerate(train_dataloader):
            #tqdm
            # 将数据移到正确的设备上（例如，GPU）
            img1, img2, intrinsics, pose = batch['img1'].to("cuda"), batch['img2'].to("cuda"), batch['intrinsic'].to(
                "cuda"), batch['motion'].to("cuda")
            input = (img1, img2, intrinsics)
            optimizer.zero_grad()  # 清零梯度

            # 执行前向传播，获取模型的输出
            _, pose_hat = model(input)

            # 从模型输出中分离平移向量和旋转向量
            T_hat = pose_hat[:, :3]  # 假设前三个数是平移向量
            R_hat = pose_hat[:, 3:]  # 假设后四个数是旋转的四元数

            # 从真实的pose中分离平移向量和旋转向量
            T = pose[:, :3]
            R = pose[:, 3:]
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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Max Rt Loss: {max_rt_loss:.4f}, Max Tra Loss: {max_tra_loss:.4f}')

        # 每训练五次保存一次模型
        if (epoch + 1) % 1 == 0:
            save_path = f'models/finetune{epoch + 1}.pth'  # 设置模型保存路径和名称
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')

        # 可以选择在训练完成后再保存一次模型
    torch.save(model.state_dict(), 'models/finetune_final.pth')
    print('Final model saved to models/finetune_final.pth')


if __name__ == '__main__':
    # 调用 main 函数
    main()