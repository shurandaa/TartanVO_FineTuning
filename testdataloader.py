from torch.utils.data import DataLoader
from Datasets.tartanTrajFlowDataset2 import TrajFolderDataset

def main():
    dataset = TrajFolderDataset(imgfolder='data/SubT_MRS_t1/image_left',
                                posefile='data/SubT_MRS_t1/ground_truth_path.csv')

    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # 迭代DataLoader并获取数据
    for i, data in enumerate(dataloader):
        # 打印出数据的形状和类型，检查是否符合预期
        print(f"Batch {i} - img1 shape: {data['img1'].shape}, img2 shape: {data['img2'].shape}")
        # 如果有标签或其他数据，也可以打印它们的形状和类型
        if 'motion' in data:
            print(f"Batch {i} - motion shape: {data['motion'].shape}
        # 如果想要在模型上测试数据
        # output = model(data['img1'], data['img2'])
        if i >= 2:  # 运行3个批次
            break

if __name__ == '__main__':
    main()