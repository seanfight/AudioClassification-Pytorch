# tensor概述
# 基本类型torch.FloatTensor ,torchTensor
# torch.LongTensor
# torch.IntTensor
# torch.ShortTensor
# torch.DoubleTensor

# tensor张量定义
import torch
# 定义0维的tensor
print(torch.tensor(2))
# 定义一维
print(torch.Tensor([1,3,6,9,0]))
# 生成tensor
# 定义一个5行3列的全为0的矩阵
print(torch.zeros(5,3,dtype=torch.int))
print(torch.rand(2,3))
# 定义未初始化
a = torch.empty(6,3)
print(torch.arange(1,14,3))
print(torch.linspace(1,9,3))
# numpy格式转换
import numpy as np
b = np.arange(1,12,3)
print(torch.from_numpy(b))
# torch维度变换
c = torch.arange(0,12,1).view(3,4)
# unsqueen 
print(c.unsqueeze(1).shape)
print(c.shape)
# 移除所有维度为1的维度
print(c.squeeze(1).shape)
# cat拼接方法
c = torch.arange(0,12,1).view(3,4)
print(c.max())
print(c.argmax())
# 