import numpy as np
# 导入dataset类
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from macls.data_utils.audio import AudioSegment

# 对一个batch的数据处理
def collate_fn(batch):
    # 找出音频长度最长的,排序
    batch = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)
    # 最长音频
    max_audio_length = batch[0][0].shape[0]
    batch_size = len(batch)
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, max_audio_length), dtype='float32')
    input_lens_ratio = []
    labels = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        labels.append(sample[1])
        seq_length = tensor.shape[0]
        # 将数据插入都0张量中，实现了padding
        inputs[x, :seq_length] = tensor[:]
        input_lens_ratio.append(seq_length/max_audio_length)
    input_lens_ratio = np.array(input_lens_ratio, dtype='float32')
    labels = np.array(labels, dtype='int64')
    return torch.tensor(inputs), torch.tensor(labels), torch.tensor(input_lens_ratio)




# 定义customdataset类
class CustomDataset(Dataset):
    def __init__(self,data_list_path):
        super(CustomDataset,self).__init__()
        self.data_list_path = data_list_path
        # 打开音频数据
        with open(data_list_path,'r',encoding='utf-8') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self,idx):
        print(self.lines[idx])
        audio_path, label = self.lines[idx].replace('\n', '').split('\t')
        audio_segment = AudioSegment.from_file(audio_path)
        # print("--------{}".format(idx))
        return np.array(audio_segment.samples, dtype=np.float32),np.array(int(label), dtype=np.int64)


# with open('learn/test_list.txt','r',encoding='utf-8') as f:
#     lines = f.readlines()
# audio_path, label = lines[0].replace('\n', '').split('\t')
# # 读取音频
# audio_segment = AudioSegment.from_file(audio_path)

# a = np.array(audio_segment.samples, dtype=np.float32),np.array(int(label), dtype=np.int64)
# print(type(a[0]))
# print(audio_segment)
# 建立第一个customdataset
customdataset_1 = CustomDataset(data_list_path = 'learn/test_list.txt')
# aa = iter(customdataset_1)
# next(aa)
# next(aa)

dataloader_1 = DataLoader(dataset=customdataset_1,collate_fn= collate_fn,batch_size = 1,shuffle=True, num_workers=4, drop_last=False)

# for i in dataloader_1:

#     print(i[2].shape[0])

a = torch.cuda.device_count()
print(a)