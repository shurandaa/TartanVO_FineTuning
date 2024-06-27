import matplotlib

matplotlib.use('TkAgg')  # 强制使用 'TkAgg' 作为后端
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 读取两个CSV文件
data1 = pd.read_csv('data/SubT_MRS_t1/groundtruthData.csv', header=None).head(4000)
data2 = pd.read_csv('data/SubT_MRS_t1/evo_using_predict/originalPredictedData.csv', header=None).head(4000)

# 创建3D图形和两个子图
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')  # 第一个子图
ax2 = fig.add_subplot(122, projection='3d')  # 第二个子图

# 初始化线条和点，设置线条宽度和点的大小
line1, = ax1.plot([], [], [], 'r-', marker='o', markersize=3, linewidth=1)  # 第一个数据集为红色线
line2, = ax2.plot([], [], [], 'b-', marker='^', markersize=3, linewidth=1)  # 第二个数据集为蓝色线


def init():
    # 初始化两个子图的界限和标签
    ax1.set_xlim(data1[1].min(), data1[1].max())
    ax1.set_ylim(data1[2].min(), data1[2].max())
    ax1.set_zlim(data1[3].min(), data1[3].max())
    ax1.set_title('Dataset 1')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')

    ax2.set_xlim(data2[1].min(), data2[1].max())
    ax2.set_ylim(data2[2].min(), data2[2].max())
    ax2.set_zlim(data2[3].min(), data2[3].max())
    ax2.set_title('Dataset 2')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_zlabel('Z Position')
    return line1, line2


def update(num):
    # 按CSV文件的顺序逐行更新数据
    # 更新第一组数据
    line1.set_data(data1.iloc[:num + 1, 1], data1.iloc[:num + 1, 2])
    line1.set_3d_properties(data1.iloc[:num + 1, 3])

    # 更新第二组数据
    line2.set_data(data2.iloc[:num + 1, 1], data2.iloc[:num + 1, 2])
    line2.set_3d_properties(data2.iloc[:num + 1, 3])

    return line1, line2


# 创建动画，确保动画更新与数据文件中的顺序相匹配
ani = FuncAnimation(fig, update, frames=min(len(data1), len(data2)), init_func=init, blit=True)

# 显示动画
plt.show()





