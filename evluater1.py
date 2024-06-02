import pandas as pd
import os


def gd_truth_processing(gd_truth_route, output_path):

    with open(gd_truth_route, 'r') as file:
        total_lines = len(file.readlines())

    gd_truth_to_process = pd.read_csv(gd_truth_route, skiprows=2, nrows=total_lines-2)

    # 生成输出文件路径，并检查是否存在相同文件名，如果存在则增加序号
    index = 1
    base_filename = "groundtruthData"
    extension = ".csv"
    output_filename = f"{base_filename}{extension}"
    final_output_path = os.path.join(output_path, output_filename)
    while os.path.exists(final_output_path):
        output_filename = f"{base_filename}{index}{extension}"
        final_output_path = os.path.join(output_path, output_filename)
        index += 1

    # 保存到新的 CSV 文件
    result_df = pd.DataFrame(gd_truth_to_process)
    result_df.to_csv(final_output_path, index=False, header=False)  # header=False 不保存列名
    print(f"Data saved to {final_output_path}")


if __name__ == '__main__':
    ground_truth_path = 'data/SubT_MRS_t1/ground_truth_path.csv'
    output_path = 'data/SubT_MRS_t1'

    gd_truth_processing(ground_truth_path, output_path)
