import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# read two CSV files
data1 = pd.read_csv('data/eURoc_DataMH4/groundtruthData.csv', header=None)
data2 = pd.read_csv('data/eURoc_DataMH4/数据/11th.csv', header=None)



# Create a 3D plot with two subplots
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')  # first subplot
ax2 = fig.add_subplot(122, projection='3d')  # second

# Set the limits and labels for the first subplot
ax1.set_xlim(data1[1].min(), data1[1].max())
ax1.set_ylim(data1[2].min(), data1[2].max())
ax1.set_zlim(data1[3].min(), data1[3].max())
ax1.set_title('Dataset 1')
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
ax1.set_zlabel('Z Position')

# Plot the first dataset
ax1.plot(data1[1], data1[2], data1[3], 'r-', marker='o', markersize=3, linewidth=1)

# second
ax2.set_xlim(data2[1].min(), data2[1].max())
ax2.set_ylim(data2[2].min(), data2[2].max())
ax2.set_zlim(data2[3].min(), data2[3].max())
ax2.set_title('Dataset 2')
ax2.set_xlabel('X Position')
ax2.set_ylabel('Y Position')
ax2.set_zlabel('Z Position')

# second
ax2.plot(data2[1], data2[2], data2[3], 'b-', marker='^', markersize=3, linewidth=1)

# show plot
plt.show()
