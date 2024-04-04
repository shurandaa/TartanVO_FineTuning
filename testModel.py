import cv2
import numpy as np
import torch
from Network.VONet import VONet  # 模型定义


print(torch.cuda.is_available())  # 应该输出 True
print(torch.cuda.get_device_name(0))  # 输出你的 GPU 名称
model = VONet()
# 加载权重
state_dict = torch.load('models/tartanvo_1914.pkl', map_location=torch.device('gpu'))  # 可以指定为 'cpu' 或你的 GPU 设备

# 检查是否使用了 DataParallel 并移除 'module.' 前缀
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# 加载修改后的权重
model.load_state_dict(new_state_dict)

print(model)