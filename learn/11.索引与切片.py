
import torch


a = torch.rand(4,3,28,28)
print(a.size())
print(a[0,0,24,24])
print(a[:,:,:,1].shape)
print(a.index_select(0,torch.tensor([0,2])))
print(a.index_select(2,torch.arange(23)).size())
print(a[:,1:,...].size())

print(a[0,...,::2].shape)

b = a.ge(0.5)
print(torch.masked_select(a,b).shape)

src = torch.tensor([
    [1,2,3],
    [4,5,6]
])
print(torch.take(src,torch.tensor([1,2,3])))

#唯独变换
# view/reshape
# squeeze/unsqueeze 删减/增加
# transpose/t/permute 转制
# expand/repeat 增加dim

a = torch.rand(4,1,28,28)
print(a.shape)
print(a.view(4,28*28).shape) 
print(a.view(4*28,28).shape)

print(a.shape)
print(a.unsqueeze(0).shape)
print(a.unsqueeze(-1).shape)

f = torch.rand(4,32,14,14)
b = torch.rand(32)
b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print(b.shape)

print(b.shape)
# print(b.squeeze(0,2,3).shape)
# print(b.expand(1,-3,1,1).shape)



# 合并与分割
# cat 拼接
# stack 拼接，创建一个新的维度 
# split 拆分 长度
# chunk 拆分 数量
a = torch.rand(4,32,8)
b = torch.rand(5,32,4)
print(b.expand(-1,-1,2).shape)
# print(torch.cat([a,b],dim=0).shape)
# print(torch.cat([a,b],dim=1).shape)