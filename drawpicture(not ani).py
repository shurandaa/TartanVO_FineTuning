import matplotlib
matplotlib.use('TkAgg')  # 强制使用 'TkAgg' 作为后端
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 读取两个CSV文件
data1 = pd.read_csv('data/eURoc_DataMH4/归一化数据/groundtruthData_aligned.csv', header=None)
data2 = pd.read_csv('data/eURoc_DataMH4/estimateDataByofficial.csv', header=None)



# 创建3D图形和两个子图
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')  # 第一个子图
ax2 = fig.add_subplot(122, projection='3d')  # 第二个子图

# 设置第一个子图的界限和标签
ax1.set_xlim(data1[1].min(), data1[1].max())
ax1.set_ylim(data1[2].min(), data1[2].max())
ax1.set_zlim(data1[3].min(), data1[3].max())
ax1.set_title('Dataset 1')
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
ax1.set_zlabel('Z Position')

# 绘制第一个数据集
ax1.plot(data1[1], data1[2], data1[3], 'r-', marker='o', markersize=3, linewidth=1)

# 设置第二个子图的界限和标签
ax2.set_xlim(data2[1].min(), data2[1].max())
ax2.set_ylim(data2[2].min(), data2[2].max())
ax2.set_zlim(data2[3].min(), data2[3].max())
ax2.set_title('Dataset 2')
ax2.set_xlabel('X Position')
ax2.set_ylabel('Y Position')
ax2.set_zlabel('Z Position')

# 绘制第二个数据集
ax2.plot(data2[1], data2[2], data2[3], 'b-', marker='^', markersize=3, linewidth=1)

# 显示图形
plt.show()
