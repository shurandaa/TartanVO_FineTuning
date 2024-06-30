import matplotlib
matplotlib.use('TkAgg')  # 强制使用 'TkAgg' 作为后端
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 读取两个CSV文件
data1 = pd.read_csv('data/eURoc_DataMH4/归一化数据/groundtruthData_aligned.csv', header=None)
data2 = pd.read_csv('data/eURoc_DataMH4/归一化数据/originalPredictedData_aligned.csv', header=None)

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 设置坐标系的界限和标签
x_min = min(data1[1].min(), data2[1].min())
x_max = max(data1[1].max(), data2[1].max())
y_min = min(data1[2].min(), data2[2].min())
y_max = max(data1[2].max(), data2[2].max())
z_min = min(data1[3].min(), data2[3].min())
z_max = max(data1[3].max(), data2[3].max())

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

ax.set_title('Comparison of Datasets')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')

# 绘制第一个数据集
ax.plot(data1[1], data1[2], data1[3], 'r-', marker='o', markersize=3, linewidth=1, label='Dataset 1')

# 绘制第二个数据集
ax.plot(data2[1], data2[2], data2[3], 'b-', marker='^', markersize=3, linewidth=1, label='Dataset 2')

# 添加图例
ax.legend()

# 显示图形
plt.show()
