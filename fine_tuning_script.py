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
import pypose as pp


def lie_algebra_loss(R_hat_quaternion, R_quaternion):
    # Calculate the rotation loss
    # as the Euclidean distance between the predicted and ground truth Lie algebra elements
    rotation_loss = torch.norm(R_hat_quaternion - R_quaternion, dim=-1).mean()

    return rotation_loss


def pose_loss_function(T_hat, T, R_hat, R, epsilon=1e-6):
    """.0
    Parameters:
     - **T_hat**: Predicted translation vector, shaped [batch_size, 3]
     - **T**: Ground truth translation vector, with the same shape as T_hat
     - **R_hat**: Predicted rotation (can be quaternion or rotation matrix), shaped [batch_size, 4] or [batch_size, 3, 3]
     - **R**: Ground truth rotation, with the same shape as R_hat
     - **epsilon**: A small constant used to avoid division by zero errors

    Returns:
     - Normalized distance loss \( L_{norm\_p} \)

    """
    # Normalize the translation vectors and calculate the Euclidean distance between the predicted and ground truth translation vectors.

    translation_loss = ((T_hat - T) ** 2).mean()
    rotation_loss = lie_algebra_loss(R_hat, R)
    # The overall loss is the translation loss
    total_loss = translation_loss
    return total_loss, rotation_loss, translation_loss


def main():
    # Define preprocessing steps
    transform = Compose([CropCenter((640, 448)),
                         DownscaleFlow(),
                         ToTensor()])

    train_dataset = TrajFolderDataset(
        imgfolder="data/targetImageFolder",
        posefile="data/eURoc_DataMH4/groundtruthSelected.csv",
        transform=transform,
        focalx=458.6539916992,  # Adjust according to your camera parameters
        focaly=457.2959899902,
        centerx=367.2149963379,
        centery=248.3750000000
    )

    # Divide the dataset into the first 900 for training, and the remaining for validation. Readers should adjust the data according to their desired training set

    train_indices = list(range(900))
    val_indices = list(range(900, len(train_dataset)))

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    train_dataloader = DataLoader(
        train_subset,
        batch_size=16,  # Adjust the batch size according to your needs
        shuffle=False,  # For predictions, it is usually not necessary to shuffle the data
        num_workers=4  # Adjust the number of worker threads according to your system configuration
    )
    val_dataloader = DataLoader(
        val_subset,
        batch_size=16,  # Adjust the batch size according to your needs
        shuffle=False,  # For predictions, it is usually not necessary to shuffle the data
        num_workers=4  # Adjust the number of worker threads according to your system configuration
    )

    # load the model
    model = TartanVO('tartanvo_1914.pkl')
    #

    for param in model.vonet.flowNet.parameters():
        param.requires_grad = False
    for param in model.vonet.flowPoseNet.parameters():
        param.requires_grad = False

        # Freeze parameter sections, only flowPoseNet.voflow_trans.parameters() is unfrozen here.
        # If you need to use both the original model and the fine-tuned model for prediction, uncomment voflow_rot.parameters() below
    for param in model.vonet.flowPoseNet.voflow_trans.parameters():
        param.requires_grad = True
    #for param in model.vonet.flowPoseNet.voflow_rot.parameters():
        #param.requires_grad = True

        # 确认冻结状态
    for name, param in model.vonet.named_parameters():
        if param.requires_grad:
            print(f"参数 {name} 将会被更新.")
        else:
            print(f"参数 {name} 已被冻结，不会被更新.")

    optimizer = torch.optim.Adam(model.vonet.parameters(), lr=0.00005, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                                 amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Learning rate scheduler unused

    # Set the total number of training epochs
    num_epochs = 200

    # Lists for recording training and validation losses
    training_log = []
    #Parameters for correcting model prediction data.
    pose_std = torch.tensor([0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=torch.float32).cuda()
    pose_std2 = np.array([0.13, 0.13, 0.13, 0.013, 0.013, 0.013], dtype=np.float32)

    for epoch in range(num_epochs):
        model.vonet.train()  # Set the model to training mode
        total_loss = 0  # Used to accumulate the loss for each epoch
        max_rt_loss = 0  # Initialize the maximum value of the rotation loss
        max_tra_loss = 0  # Initialize the maximum value of the translation loss

        # Iterate over all batches in the training data loader
        for batch_idx, batch in enumerate(train_dataloader):
            # tqdm
            # Move the data to the appropriate device (e.g., GPU)
            img1, img2, intrinsics, pose = batch['img1'].to("cuda"), batch['img2'].to("cuda"), batch['intrinsic'].to(
                "cuda"), batch['motion'].to("cuda")
            motion = batch['motion']
            input = [img1, img2, intrinsics]
            optimizer.zero_grad()

            _, pose_hat = model.vonet(input)
            # Separate the translation vector and rotation vector from the model output
            pose_hat = pose_hat * pose_std
            scale = torch.norm(motion[:, :3], dim=1).to("cuda")
            trans_est = pose_hat[:, :3]
            trans_est_normalized = trans_est / torch.norm(trans_est, dim=1).view(-1, 1) * scale.view(-1, 1)

            pose_hat_corrected = torch.cat([trans_est_normalized, pose_hat[:, 3:]], dim=1)

            T_hat = pose_hat_corrected[:, :3]
            R_hat = pose_hat_corrected[:, 3:]
            # Separate the translation vector and rotation vector from the ground truth poses,
            # and ensure tensors are created on the GPU
            T = pose[:, :3].to(dtype=torch.float32).to("cuda", non_blocking=True)
            T.requires_grad_()
            R = pose[:, 3:].to(dtype=torch.float32).to("cuda", non_blocking=True)
            R.requires_grad_()

            # Calculate the loss using the custom loss function
            loss, rt_loss, tra_loss = pose_loss_function(T_hat, T, R_hat, R)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Accumulate the loss
            if rt_loss.item() > max_rt_loss:
                max_rt_loss = rt_loss.item()
            if tra_loss.item() > max_tra_loss:
                max_tra_loss = tra_loss.item()

        # Calculate the average loss for this epoch
        avg_loss = total_loss / len(train_dataloader)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Max Rt Loss: {max_rt_loss:.4f}, Max Tra Loss: {max_tra_loss:.4f}')
        # scheduler.step()

        val_totalloss = 0
        model.vonet.eval()
        with torch.no_grad():  # Disable gradient calculation
            for batch_idx, batch in enumerate(val_dataloader):
                img1, img2, intrinsics, pose = batch['img1'].to("cuda"), batch['img2'].to("cuda"), batch[
                    'intrinsic'].to(
                    "cuda"), batch['motion'].to("cuda")
                motion2 = batch['motion']
                input = [img1, img2, intrinsics]

                _, pose_hatval = model.vonet(input)
                pose_hatval = pose_hatval.data.cpu().numpy()
                pose_hatval = pose_hatval * pose_std2  # The output is normalized during training, now scale it back
                scale = np.linalg.norm(motion2[:, :3], axis=1)
                trans_est = pose_hatval[:, :3]
                trans_est = trans_est / np.linalg.norm(trans_est, axis=1).reshape(-1, 1) * scale.reshape(-1, 1)
                pose_hatval[:, :3] = trans_est
                # Separate the translation vector and rotation vector from the model output
                T_hat = torch.tensor(pose_hatval[:, :3], dtype=torch.float32, device="cuda")
                R_hat = torch.tensor(pose_hatval[:, 3:], dtype=torch.float32, device="cuda")
                # Separate the translation vector and rotation vector from the ground truth poses
                T = pose[:, :3].to(dtype=torch.float32).to("cuda", non_blocking=True)
                R = pose[:, 3:].to(dtype=torch.float32).to("cuda", non_blocking=True)

                # Calculate the loss using the custom loss function
                loss, rt_loss, tra_loss = pose_loss_function(T_hat, T, R_hat, R)
                val_totalloss += loss.item()

        val_avg_loss = val_totalloss / len(val_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_avg_loss:.4f}')

        # Record the training and validation losses
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_loss': val_avg_loss
        })

        # Save the model after each training session
        if (epoch + 1) % 1 == 0:
            save_path = f'models/解冻pose与rot并将二模型结合/finetuneEuroc{epoch + 1}.pkl'  # 设置模型保存路径和名称
            torch.save(model.vonet.state_dict(), save_path)
            print(f'Model saved to {save_path}')

        # Optionally, save the model again after the training is completed
    torch.save(model.vonet.state_dict(), 'models/解冻pose与rot并将二模型结合/finetune_final.pkl')
    print('Final model saved ')

    # Save the training and validation losses to a CSV file
    df = pd.DataFrame(training_log)
    df.to_csv('models/解冻pose与rot并将二模型结合/training_log.csv', index=False)


if __name__ == '__main__':
    # Call the main function
    main()