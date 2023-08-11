# 
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
# batch_size每次处理的图片数量
batch_size = 512
# step1,load_dataset
from torch.utils.data import DataLoader

from visualdl import LogWriter
import cv2
# 定义编码函数
def one_hot(label,depth=10):
    out = torch.zeros(label.size(0),depth)
    idx = torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1,index=idx,value=1)
    return out
train_lpoader = DataLoader(
    torchvision.datasets.MNIST('mnist_data',train=True,download=True,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    # 平均值和标准差归一化图片
                                    # 标准化：transforms.Normalize
                                    # torchvision.transforms.Normalize(mean, std)
                                    # 用平均值和标准偏差归一化张量图像。给定mean：(M1,…,Mn)和std：(S1,…,Sn)对于n通道，此变换将标准化输入的每个通道，torch.*Tensor即 input[channel] = (input[channel] - mean[channel]) / std[channel]
                                    # mean（sequence） - 每个通道的均值序列。
                                    # std（sequence） - 每个通道的标准偏差序列。

                                    torchvision.transforms.Normalize(
                                        (0.1307,),(0.3081,))
                                    ])
                                ),
                                # shuffle 对数据进行重新排序
                                batch_size = batch_size,shuffle = True
)

test_lpoader = DataLoader(
    torchvision.datasets.MNIST('mnist_data',train=True,download=True,
                               transform=torchvision.transforms.Compose(
                                   [torchvision.transforms.ToTensor(),
                                    # 平均值和标准差归一化图片
                                    # 标准化：transforms.Normalize
                                    # torchvision.transforms.Normalize(mean, std)
                                    # 用平均值和标准偏差归一化张量图像。给定mean：(M1,…,Mn)和std：(S1,…,Sn)对于n通道，此变换将标准化输入的每个通道，torch.*Tensor即 input[channel] = (input[channel] - mean[channel]) / std[channel]
                                    # mean（sequence） - 每个通道的均值序列。
                                    # std（sequence） - 每个通道的标准偏差序列。

                                    torchvision.transforms.Normalize(
                                        (0.1307,),(0.3081,))
                                    ])
                                ),
                                # shuffle 对数据进行重新排序
                                batch_size = batch_size,shuffle = True
)

x,y = next(iter(train_lpoader))
print(x.shape,y.shape,x.min(),x.max())
# torch.Size([512, 1, 28, 28]) torch.Size([512]) tensor(-0.4242) tensor(2.8215)


# 创建网络
class Net(nn.Module):
    # 初始化init
    def __init__(self):
        # 初始化父类
        super(Net,self).__init__()
        # 第一层，线性层
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self,x):
        # x:[b,1,28,28]
        # h = xw+b
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = self.fc3(x)
        return x
# net init
net = Net()
# 优化器
# [w1,b1,w2,b2,w3,b3]
optimizer = optim.SGD(net.parameters(),lr = 0.01,momentum=0.9)
# 初始化一个记录器
# with LogWriter(logdir='./train') as writer
writer = LogWriter(logdir = 'learn/train')
step = 0
for epoch in range(3):
    for batch_idx,(x,y) in enumerate(train_lpoader):
        # print("////")
        # print(x.shape,y.shape)
        # x[b,1,28,28] >> x[b,784]
        x = x.view(x.size(0),28*28)
        out = net(x)
        y_onehot = one_hot(y)
        # 计算梯度值
        loss = F.mse_loss(out,y_onehot)
        # 清空梯度
        optimizer.zero_grad()
        loss.backward()
        # w' = w-lr*grad
        optimizer.step()
        # print(batch_idx)
        if batch_idx % 10== 0:
            print(epoch,batch_idx,loss.item())

        step = batch_idx + step
        # print(step,loss.item())
        writer.add_scalar(tag = "n",value = loss.item(),step=step)


# 准确度测试
total_correct = 0
for x,y in test_lpoader:
    x = x.view(x.size(0),28*28)
    out = net(x)
    pread = out.argmax(dim=1)
    correct = pread.eq(y).sum().float()
    total_correct += correct


    
    # writer.add_image(tag = 'a',img = x[0],step=1)
# total_num = len(test_lpoader.dataset)
# acc = total_correct/total_num
# print(acc)
x,y = next(iter(test_lpoader))
out = net(x.view(x.size(0),28*28))
pread = out.argmax(dim=1)
for i in range(10):
    a = x[i].view(28,28,1)*0.3081+0.1307
    # writer.add_image(tag = str(pread[i].item()),img = a.numpy(),step=i)
    writer.add_image(tag = 'g',img = a.numpy(),step=i)
