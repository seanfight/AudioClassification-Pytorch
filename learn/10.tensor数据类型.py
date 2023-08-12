# all is about Tensor
import torch
import numpy as np
# pytorch
# torch.FloatTensor     torch.cuda.FloatTensor
# torch.DoubleTensor    torch.cuda.DoubleTensor
    # tensor = torch.ByteTensor([1,2])
    # print(tensor)
    # print(tensor.type())
    # print(tensor.dtype)
    # print(tensor.is_cuda)
# torch.ByteTensor      torch.cuda.ByteTensor
# torch.CharTensor      torch.cuda.CharTensor
# torch.IntTensor       torch.cuda.IntTensor
# torch.LongTensor      torch.cuda.LongTensor
a = torch.randn(2,3)
print(isinstance(a,torch.FloatTensor))
print(isinstance(a,torch.cuda.FloatTensor))
# 查询是否有gpu
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# # 查看gpu名字
# print(torch.cuda.get_device_name(0))
# # 当前设备索引
# print(torch.cuda.current_device())

# rank/dim = 0,标量
print(torch.tensor(1.))
print(torch.tensor(1.3))

a = torch.tensor(2)
b = torch.tensor([2,3])
c = torch.tensor([1.1,2.1,3.1])
print(a.type(),b.type())
print(a.shape,b.shape,c.shape)
print(a.size())

print(torch.tensor([1.1]))
print(c)
print(torch.Tensor(1).shape)
print(torch.ones(2))
print(torch.ones(2,2))
print(torch.ones(2,2,2).shape)
 
a =np.array([1.,2.,3.])
a = torch.from_numpy(a)
# print(torch.from_numpy(a))
print(torch.tensor([1,2,3]))
print(torch.Tensor([1,2,3]))
print(torch.empty(2,3))
print(torch.empty([2,3]))
print(torch.FloatTensor(2,3))
print(torch.empty(2).type())
# 随机初始化
# print(torch.rand(3.,3.))
print(a)
print(torch.rand_like(a))
print(torch.randint(1,10,[3,3]))
# normal
print(torch.full([2,3],4))
print(torch.full([],4).item())
print(torch.arange(0,10,2))
print(torch.linspace(0,10,steps=10))
print(torch.logspace(0,1,2))
print(torch.eye(2,4))