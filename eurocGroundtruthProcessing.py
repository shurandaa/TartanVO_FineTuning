import pandas as pd


def read_and_swap_columns(csv_file_path, output_file_path):
    # 读取CSV文件的前8列
    df = pd.read_csv(csv_file_path, usecols=range(8))

    # 打印原始数据
    print("原始数据:")
    print(df.head())

    # 将第5列和第8列调换
    df.iloc[:, [4, 7]] = df.iloc[:, [7, 4]].values

    # 打印调换后的数据
    print("调换第5列和第8列后的数据:")
    print(df.head())

    # 保存处理后的数据到新的CSV文件
    df.to_csv(output_file_path, index=False, header=False)


# 示例用法
csv_file_path = 'data/eURoc_DataMH4/groundtruthSelected.csv'  # 输入CSV文件路径
output_file_path = 'data/eURoc_DataMH4/groundtruthData.csv'  # 输出CSV文件路径
read_and_swap_columns(csv_file_path, output_file_path)
